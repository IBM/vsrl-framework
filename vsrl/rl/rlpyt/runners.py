#
# Copyright (C) 2020 IBM. All Rights Reserved.
#
# See LICENSE.txt file in the root directory
# of this source tree for licensing information.
#

import math
from dataclasses import dataclass, field
from time import time
from typing import Any, Dict, Optional

import psutil
import torch
from rlpyt.runners.base import BaseRunner
from rlpyt.utils.seed import make_seed, set_seed
from tqdm.auto import tqdm

from .loggers import BaseLogger


@dataclass
class MinibatchRl(BaseRunner):
    """
    Runs RL on minibatches; tracks performance with a Logger.

    This also supports additional samplers for evaluation.
    """

    algo: Any
    agent: Any
    sampler: Any
    n_steps: int
    seed: int = field(default_factory=make_seed)
    affinity: Dict[str, Any] = field(
        default_factory=dict
    )  # Hardware component assignments for sampler and algorithm
    log_interval_steps: int = 2_000  # Number of environment steps between logging
    n_runners: int = 1
    n_eval_steps: int = 0
    logger: BaseLogger = field(default_factory=BaseLogger)
    itr_batch_size: int = field(init=False)
    n_itr: int = field(init=False)
    pbar: bool = True
    extra_eval_samplers: Dict[str, Any] = field(default_factory=dict)
    state_dict_fname: Optional[str] = None
    eval_only: bool = False
    start_itr: int = 0

    def __post_init__(self):
        self.itr_batch_size = self.sampler.batch_spec.size * self.n_runners
        self.n_eval_itr = math.ceil(self.n_eval_steps / self.itr_batch_size)
        self.n_eval_steps = self.n_eval_itr * self.itr_batch_size
        self.min_itr_learn = getattr(self.algo, "min_itr_learn", 0)
        self.log_itrs = 0  # set during initialize

    def startup(self):
        """
        Sets hardware affinities, initializes the following: 1) sampler (which
        should initialize the agent), 2) agent device and data-parallel wrapper (if applicable),
        3) algorithm, 4) logger.

        This function is nearly identical to MinibatchRlBase.startup with the main
        difference being the initialization of the extra eval samplers.
        """
        p = psutil.Process()
        try:
            if self.affinity.get("master_cpus", None) is not None and self.affinity.get(
                "set_affinity", True
            ):
                p.cpu_affinity(self.affinity["master_cpus"])
            cpu_affin = p.cpu_affinity()
        except AttributeError:
            cpu_affin = "UNAVAILABLE MacOS"
        print(
            f"Runner {getattr(self, 'rank', '')} master CPU affinity: " f"{cpu_affin}."
        )
        if self.affinity.get("master_torch_threads", None) is not None:
            torch.set_num_threads(self.affinity["master_torch_threads"])
        print(
            f"Runner {getattr(self, 'rank', '')} master Torch threads: "
            f"{torch.get_num_threads()}."
        )
        set_seed(self.seed)
        self.rank = rank = getattr(self, "rank", 0)
        self.world_size = world_size = getattr(self, "world_size", 1)

        for i, sampler in enumerate(self.extra_eval_samplers.values()):
            sampler.initialize(
                agent=self.agent,  # Agent gets intialized in sampler.
                affinity=self.affinity,
                seed=self.seed + i,
                bootstrap_value=getattr(self.algo, "bootstrap_value", False),
                traj_info_kwargs=self.get_traj_info_kwargs(),
                rank=rank,
                world_size=world_size,
            )
        examples = self.sampler.initialize(
            agent=self.agent,  # Agent gets intialized in sampler.
            affinity=self.affinity,
            seed=self.seed + 1,
            bootstrap_value=getattr(self.algo, "bootstrap_value", False),
            traj_info_kwargs=self.get_traj_info_kwargs(),
            rank=rank,
            world_size=world_size,
        )
        self.itr_batch_size = self.sampler.batch_spec.size * world_size
        n_itr = self.get_n_itr()
        self.agent.to_device(self.affinity.get("cuda_idx", None))
        if world_size > 1:
            self.agent.data_parallel()
        self.algo.initialize(
            agent=self.agent,
            n_itr=self.n_itr,
            batch_spec=self.sampler.batch_spec,
            mid_batch_reset=self.sampler.mid_batch_reset,
            examples=examples,
            world_size=world_size,
            rank=rank,
        )
        print(f"Running {self.n_itr} iterations with batch size {self.itr_batch_size}.")
        self.logger.initialize_logging()
        return n_itr

    def get_traj_info_kwargs(self):
        """
        Pre-defines any TrajInfo attributes needed from elsewhere e.g.
        algorithm discount factor.
        """
        return dict(discount=getattr(self.algo, "discount", 1))

    def get_n_itr(self):
        """
        Determine number of train loop iterations to run.  Converts logging
        interval units from environment steps to iterations.
        """
        # Log at least as often as requested (round down itrs):
        log_itrs = max(self.log_interval_steps // self.itr_batch_size, 1)
        n_itr = int(math.ceil(self.n_steps / self.itr_batch_size))
        if n_itr % log_itrs > 0:  # Keep going to next log itr.
            n_itr += log_itrs - (n_itr % log_itrs)
        self.log_itrs = log_itrs
        self.n_itr = n_itr
        print(f"Running {n_itr} iterations of minibatch RL.")
        return n_itr

    def shutdown(self, error: bool = False):
        print("Training complete.")
        self.sampler.shutdown()
        for sampler in self.extra_eval_samplers.values():
            sampler.shutdown()
        self.logger.shutdown(error)

    def get_itr_snapshot(self, itr):
        """
        Returns all state needed for full checkpoint/snapshot of training run,
        including agent parameters and optimizer parameters.
        """
        return dict(
            itr=itr,
            cum_steps=itr * self.sampler.batch_size * self.world_size,
            agent_state_dict=self.agent.state_dict(),
            optimizer_state_dict=self.algo.optim_state_dict(),
        )

    def save_itr_snapshot(self, itr, metric: Optional[float] = None):
        """
        Calls the logger to save training checkpoint/snapshot (logger itself
        may or may not save, depending on mode selected).
        """
        params = self.get_itr_snapshot(itr)
        self.logger.save_itr_params(itr, params, metric)

    def train(self):
        """
        Performs startup, evaluates the initial agent, then loops by
        alternating between ``sampler.obtain_samples()`` and
        ``algo.optimize_agent()``.  Pauses to evaluate the agent at the
        specified log interval.
        """
        n_itr = self.startup()
        if self.state_dict_fname:
            state_dict = torch.load(self.state_dict_fname)
            self.agent.load_state_dict(state_dict["agent_state_dict"])
            # TODO - load optimizer state dict. where is the optimizer? in algo?

        if self.eval_only:
            self.eval_loop()

        desc = f"{self.sampler.EnvCls.__name__}-{self.algo.__class__.__name__}"
        if self.pbar:
            pbar = tqdm(total=n_itr * self.itr_batch_size, unit_scale=True, desc=desc)
        shutdown = False
        try:
            for itr in range(self.start_itr, n_itr):
                self.agent.sample_mode(itr)
                samples, traj_infos = self.sampler.obtain_samples(itr)
                self.agent.train_mode(itr)
                opt_info = self.algo.optimize_agent(itr, samples)
                if itr % self.log_itrs == 0:
                    self.logger.log(itr, traj_infos, opt_info)
                else:
                    self.logger.log_fast(itr, traj_infos, opt_info)

                if self.n_eval_itr and (itr + 1) % self.n_eval_itr == 0:
                    self.agent.eval_mode(itr)
                    start = time()
                    # TODO: have self.eval_samplers be the normal sampler + the extra
                    # ones? to slightly simplify this part
                    for name, sampler in self.extra_eval_samplers.items():
                        traj_infos = sampler.evaluate_agent(itr)
                        traj_infos = [
                            {f"{name}_{key}": val for key, val in traj_info.items()}
                            for traj_info in traj_infos
                        ]
                        self.logger.log(itr, traj_infos, test=True)
                    traj_infos = self.sampler.evaluate_agent(itr)
                    self.logger.log(itr, traj_infos, test=True)
                    self.logger.log_metric("eval_time", time() - start)

                    avg_return = (
                        sum(t["Return"] for t in traj_infos) / len(traj_infos)
                        if traj_infos
                        else float("nan")
                    )
                    self.save_itr_snapshot(itr, avg_return)

                    if self.pbar:
                        pbar.set_description(desc + f"({avg_return:,.3f})")
                if self.pbar:
                    pbar.update(self.itr_batch_size)
            self.shutdown()
            shutdown = True
        finally:
            if self.pbar:
                pbar.close()
            if not shutdown:
                self.shutdown(error=True)

    def eval_loop(self):
        """
        This can be used for checking how well constraints work by just evaluating the
        agent repeatedly. The Collector (rlpyt class) should handle saving unsafe
        trajectories or other desired data.
        """
        itr = 0
        self.agent.eval_mode(itr)
        pbar = tqdm()

        all_samplers = [self.sampler] + list(self.extra_eval_samplers.values())

        try:
            while True:
                for sampler in all_samplers:
                    sampler.evaluate_agent(itr)
                pbar.update(1)
        finally:
            self.shutdown()
            pbar.close()
