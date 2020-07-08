#
# Copyright (C) 2020 IBM. All Rights Reserved.
#
# See LICENSE.txt file in the root directory
# of this source tree for licensing information.
#

from pathlib import Path
from typing import Optional

import comet_ml  # don't remove this
import torch

from vsrl.rl import envs
from vsrl.rl.rlpyt.agents import (
    SafeCategoricalPgAgent,
    SafeGaussianPgAgent,
    SafeSacAgent,
)
from vsrl.rl.rlpyt.algos import PPO, SAC
from vsrl.rl.rlpyt.env import RLPytEnv, SafetyEnvTrajInfo
from vsrl.rl.rlpyt.loggers import BaseLogger, CometLogger
from vsrl.rl.rlpyt.models import (
    ImpalaSacModel,
    ImpalaVisModel,
    OracleModel,
    VisionModel,
    VisionModel2,
)
from vsrl.rl.rlpyt.runners import MinibatchRl
from vsrl.symmap.detectors.center_track import CenterTrack, ResNetCT, ResNetCT0UC_SH


# max errors:
# ACC: 3
# PMGoalFinding: 4
# Pointmesses: 4
def run(
    env_name: str = "ACC",
    algo: str = "PPO",
    detector_path: Optional[str] = None,
    detector_max_error: int = 0,
    safe: bool = False,
    unsafe_reward: Optional[float] = None,
    n_steps: int = 2_000_000,
    snapshot_gap: int = 1,
    eval_n_envs: int = 5,
    eval_max_steps: int = 1000,  # max steps *per env*
    eval_max_trajectories: int = 5,
    n_steps_between_eval: int = 10_000,
    entropy_coeff: float = 0.01,
    use_logger: bool = True,
    exp_set: Optional[str] = None,
    detector: Optional[str] = None,
    snapshot_dir: str = "",
    state_dict_fname: Optional[str] = None,
    pbar: bool = True,
    eval_only: bool = False,
    use_print_exp: bool = False,
    oracle_obs: bool = False,
    debug: bool = False,
    device: int = -1,
    n_cpus: int = 1,
    cpu_offset: int = 0,
    continuous: bool = True,
    batch_T: int = 64,
    batch_B: int = 32,
    ppo_epochs: int = 4,
    ppo_minibatches: int = 4,
    learning_rate: float = 1e-3,
    walls: bool = False,
    dense_rewards: bool = False,
    vis_model: str = "VisionModel",
    img_scale: int = 4,
    grayscale: bool = True,
):
    if detector_path is not None:
        safe = True
    if safe and not oracle_obs and detector_path is None:
        raise ValueError(
            "detector_path must be given to be safe (without oracle observations)."
        )
    run_kwargs = locals().copy()

    algo_configs = dict(
        PPO=dict(
            discount=0.99,
            learning_rate=learning_rate,
            value_loss_coeff=1.0,
            entropy_loss_coeff=0.01,
            clip_grad_norm=1.0,
            gae_lambda=0.98,
            linear_lr_schedule=True,
            minibatches=ppo_minibatches,
            epochs=ppo_epochs,
        ),
        SAC=dict(
            bootstrap_timelimit=False,  # should maybe be true, but have to pass horizon properly then
        ),
    )

    config = dict(
        agent=dict(),
        model=dict(),
        optim=dict(),
        runner=dict(),
        algo=algo_configs[algo],
        sampler=dict(batch_T=batch_T, batch_B=batch_B, max_decorrelation_steps=1000),
    )

    if vis_model == "VisionModel2":
        VisModel = VisionModel2
    elif "impala" in vis_model.lower():
        vis_model = "ImpalaVisModel"
        VisModel = ImpalaVisModel
    else:
        vis_model = "VisionModel"
        VisModel = VisionModel

    if oracle_obs:
        config["agent"]["ModelCls"] = OracleModel
    else:
        config["agent"]["ModelCls"] = VisModel

    if debug:
        config["sampler"]["batch_T"] = 16
        config["sampler"]["batch_B"] = 2
        config["sampler"]["max_decorrelation_steps"] = 10

    if safe:
        # if we want to decorrelate safely, we need to get the sym features there too
        config["sampler"]["max_decorrelation_steps"] = 0

    if debug or device == -1:
        print("Running serially on the CPU")
        parallel = False
        # parallel = True
        from vsrl.rl.rlpyt.collectors import SafeCpuResetCollector as Collector
        from vsrl.rl.rlpyt.collectors import SafeSerialEvalCollector as EvalCollector

        if parallel:
            from rlpyt.samplers.parallel.cpu.sampler import CpuSampler as Sampler
        else:
            from rlpyt.samplers.serial.sampler import SerialSampler as Sampler
    else:
        # on some machines, using a subprocess to get the sample buffers is hanging
        use_subprocess = False
        if not use_subprocess:
            from rlpyt.samplers.buffer import build_samples_buffer
            from rlpyt.samplers.parallel.base import ParallelSamplerBase

            def _build_buffers(self, env, bootstrap_value):
                self.samples_pyt, self.samples_np, examples = build_samples_buffer(
                    self.agent,
                    env,
                    self.batch_spec,
                    bootstrap_value,
                    agent_shared=True,
                    env_shared=True,
                    subprocess=False,
                )
                return examples

            ParallelSamplerBase._build_buffers = _build_buffers

        assert torch.cuda.is_available(), "cuda must be available if device >= 0"
        print("Running in parallel on the GPU")
        from vsrl.rl.rlpyt.collectors import SafeGpuResetCollector as Collector
        from vsrl.rl.rlpyt.collectors import SafeGpuEvalCollector as EvalCollector
        from rlpyt.samplers.parallel.gpu.sampler import GpuSampler as Sampler

        # from rlpyt.samplers.parallel.gpu.alternating_sampler import AlternatingSampler as Sampler

    # TODO - make penalization an option (should be in env_kwargs now)
    # if safe:
    # assert unsafe_reward is not None
    # safety_wrapper_kwargs = {
    # "unsafe_reward": unsafe_reward,
    # }

    if detector_path is not None:
        detector = CenterTrack.load_from_checkpoint(detector_path)
        if grayscale is not detector.grayscale:
            raise ValueError(
                f"The given detector has grayscale={detector.grayscale} but you wanted "
                f"the environment to have grayscale={grayscale}."
            )

    base_env = getattr(envs, env_name)(
        grayscale=grayscale, img_scale=img_scale, oracle_obs=oracle_obs
    )
    test_obs = base_env.reset()
    model_kwargs = {"categorical": not continuous}
    if not oracle_obs:
        model_kwargs["n_vector_obs"] = test_obs.vector.size

    if continuous:
        Agent = SafeGaussianPgAgent
        model_kwargs["action_dim"] = len(base_env.action_space.bounds)
    else:
        Agent = SafeCategoricalPgAgent
        model_kwargs["action_dim"] = len(base_env.action_space.elements)
    agent_kwargs = {}

    # TODO - just have models take obs_shape? and get it as env.reset().shape?
    # see BaseAgent for how you can easily get the spaces
    if oracle_obs:
        model_kwargs["n_inputs"] = base_env._state.size
    else:
        model_kwargs["img_shape"] = (
            base_env._height,
            base_env._width,
            2,
            # 1 if base_env._grayscale else 3,
        )
        if safe:
            model_kwargs["sym_extractor"] = base_env.SymFeatClass(detector)

    make_env = lambda *args, **kwargs: RLPytEnv(
        vsrl_env=getattr(envs, env_name)(
            grayscale=grayscale,
            img_scale=img_scale,
            oracle_obs=oracle_obs,
            safe_sep=detector_max_error + 1,
            walls=walls,
            dense_rewards=dense_rewards,
        ),
        log_unsafe_transitions=debug,
    )

    model_kwargs.update(config["model"])

    if algo == "PPO":
        config["algo"]["entropy_loss_coeff"] = entropy_coeff
        Algo = PPO
    elif algo == "SAC":
        Agent = SafeSacAgent
        Algo = SAC
        config["agent"]["ModelCls"] = ImpalaSacModel
        vis_model = "ImpalaSacModel"
        model_kwargs.pop("action_dim")
        model_kwargs.pop("img_shape")
    else:
        raise ValueError(f"Algorithm {algo} is not supported.")

    algo = Algo(optim_kwargs=config["optim"], **config["algo"])
    agent = Agent(model_kwargs=model_kwargs, **config["agent"], **agent_kwargs)

    config["runner"]["n_steps"] = n_steps
    config["runner"]["n_eval_steps"] = n_steps_between_eval
    config["runner"]["state_dict_fname"] = state_dict_fname
    config["sampler"].update(
        {
            "eval_n_envs": eval_n_envs,
            "eval_max_steps": eval_max_steps * eval_n_envs,
            "eval_max_trajectories": eval_max_trajectories,
        }
    )
    config["logger"] = {
        "exp_set": exp_set,
        "snapshot_dir": snapshot_dir,
        # "snapshot_gap": snapshot_gap,
        "use_print_exp": use_print_exp,
    }

    affinity = {"workers_cpus": list(range(cpu_offset, cpu_offset + n_cpus))}
    if device >= 0:
        affinity["cuda_idx"] = device

    batch_size = config["sampler"]["batch_T"] * config["sampler"]["batch_B"]
    if use_logger:
        logger = CometLogger(
            batch_size,
            **config["logger"],
            log_env_gpu=False,
            log_env_cpu=False,
            auto_output_logging=not pbar,
            disabled=debug,
        )
        logger.log_config(config)
        logger.log_parameters(run_kwargs)
        logger.log_parameters(
            {
                "grayscale": grayscale,
                "img_scale": img_scale,
                "oracle_obs": oracle_obs,
                "continuous": continuous,
                "vis_model": vis_model,
                "walls": walls,
                "dense_rewards": dense_rewards,
                "algo": algo,
            }
        )
    else:
        logger = BaseLogger()

    Collector.safe = safe
    EvalCollector.safe = safe
    sampler = Sampler(
        EnvCls=make_env,
        env_kwargs={},
        eval_env_kwargs={},
        CollectorCls=Collector,
        eval_CollectorCls=EvalCollector,
        TrajInfoCls=SafetyEnvTrajInfo,
        **config["sampler"],
    )

    runner = MinibatchRl(
        algo=algo,
        agent=agent,
        sampler=sampler,
        affinity=affinity,
        logger=logger,
        pbar=pbar,
        eval_only=eval_only,
        **config["runner"],
    )

    if debug:
        with torch.autograd.set_detect_anomaly(True):
            runner.train()
    else:
        runner.train()

    return runner
