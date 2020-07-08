#
# Copyright (C) 2020 IBM. All Rights Reserved.
#
# See LICENSE.txt file in the root directory
# of this source tree for licensing information.
#

import json
from contextlib import nullcontext
from pathlib import Path
from time import time
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
from rlpyt.utils.logging.logger import disable as disable_default_logger

disable_default_logger()


class BaseLogger:
    def save_itr_params(
        self, step: int, params: Dict[str, Any], metric: Optional[float] = None
    ) -> None:
        pass

    def initialize_logging(self):
        pass

    def shutdown(self, error: bool = False):
        pass

    def log_fast(
        self,
        step: int,
        traj_infos: Sequence[Dict[str, float]],
        opt_info: Optional[Tuple[Sequence[float], ...]] = None,
        test: bool = False,
    ) -> None:
        """Called after every iteration; only log / store essential things."""

    def log(
        self,
        step: int,
        traj_infos: Sequence[Dict[str, float]],
        opt_info: Optional[Tuple[Sequence[float], ...]] = None,
        test: bool = False,
    ):
        """
        Called every log_itrs iterations; log everything.

        This should call log_fast, so don't use both of them on the same iteration.
        """

    def log_metric(self, name, val):
        pass

    def log_config(self, config):
        pass


class Logger(BaseLogger):
    def __init__(
        self,
        batch_size: int,
        snapshot_dir: Optional[str] = None,
        snapshot_mode: str = "best",
        snapshot_gap: int = 1,
    ):
        """
        :param snapshot_dir: The directory where snapshots of the model parameters should
          be saved. If None, no snapshots are saved.
        :param snapshot_mode: one of {all, last, best}.
          * all: save a new snapshot every time `save_itr_params` is called.
          * last: only save one snapshot; overwrite it at each call.
          * best: only save one snapshot; overwrite if model improves
          The snapshot files will be named "snapshot_{step}.torch" where step is the current
          step.
        """
        self.batch_size = batch_size
        self.snapshot_dir = Path(snapshot_dir) if snapshot_dir else None
        self.snapshot_mode = snapshot_mode
        self.snapshot_gap = snapshot_gap
        self._previous_snapshot_fname = ""
        self._best_metric = -float("inf")

        if self.snapshot_dir:
            self.snapshot_dir.mkdir(parents=True, exist_ok=True)

    def save_itr_params(
        self, step: int, params: Dict[str, Any], metric: Optional[float] = None
    ) -> None:
        if not self.snapshot_dir or (step + 1) % self.snapshot_gap != 0:
            return

        if self.snapshot_mode == "best":
            if metric is None or metric < self._best_metric:
                return
            self._best_metric = metric

        snapshot_fname = self.snapshot_dir / f"snapshot_{step}.torch"
        torch.save(params, snapshot_fname)

        if self.snapshot_mode != "all":
            # remove previous file after new one is created
            if self._previous_snapshot_fname:
                Path(self._previous_snapshot_fname).unlink()
            self._previous_snapshot_fname = snapshot_fname


# TODO - check if comet slows things down much. If so... can we speed up somehow?
# by storing locally instead of uploading right away or by batching uploads somehow?
# Could even raise an issue on GitHub if it's much of a problem and ask if it could be
# made async somehow?
class CometLogger(Logger):
    def __init__(
        self,
        batch_size: int,
        snapshot_dir: Optional[str] = None,
        snapshot_mode: str = "last",
        snapshot_gap: int = 1,
        exp_set: Optional[str] = None,
        use_print_exp: bool = False,
        saved_exp: Optional[str] = None,
        **kwargs,
    ):
        """
        :param kwargs: passed to comet's Experiment at init.
        """
        if use_print_exp:
            self.experiment = PrintExperiment()
        else:
            from comet_ml import Experiment, ExistingExperiment, OfflineExperiment

            if saved_exp:
                self.experiment = ExistingExperiment(
                    previous_experiment=saved_exp, **kwargs
                )
            else:
                try:
                    self.experiment = Experiment(**kwargs)
                except ValueError:  # no API key
                    log_dir = Path.home() / "logs"
                    log_dir.mkdir(exist_ok=True)
                    self.experiment = OfflineExperiment(offline_directory=str(log_dir))

        self.experiment.log_parameter("complete", False)
        if exp_set:
            self.experiment.log_parameter("exp_set", exp_set)
        if snapshot_dir:
            snapshot_dir = Path(snapshot_dir) / self.experiment.get_key()
        # log_traj_window (int): How many trajectories to hold in deque for computing performance statistics.
        self.log_traj_window = 100
        self._cum_metrics = {
            "n_unsafe_actions": 0,
            "constraint_used": 0,
            "cum_completed_trajs": 0,
            "logging_time": 0,
        }
        self._new_completed_trajs = 0
        self._last_step = 0
        self._start_time = self._last_time = time()
        self._last_snapshot_upload = 0
        self._snaphot_upload_time = 30 * 60

        super().__init__(batch_size, snapshot_dir, snapshot_mode, snapshot_gap)

    def log_fast(
        self,
        step: int,
        traj_infos: Sequence[Dict[str, float]],
        opt_info: Optional[Tuple[Sequence[float], ...]] = None,
        test: bool = False,
    ) -> None:
        if not traj_infos:
            return
        start = time()

        self._new_completed_trajs += len(traj_infos)
        self._cum_metrics["cum_completed_trajs"] += len(traj_infos)
        # TODO: do we need to support sum(t[k]) if key in k?
        # without that, this doesn't include anything from extra eval samplers
        for key in self._cum_metrics:
            if key == "cum_completed_trajs":
                continue
            self._cum_metrics[key] += sum(t.get(key, 0) for t in traj_infos)
        self._cum_metrics["logging_time"] += time() - start

    def log(
        self,
        step: int,
        traj_infos: Sequence[Dict[str, float]],
        opt_info: Optional[Tuple[Sequence[float], ...]] = None,
        test: bool = False,
    ):
        self.log_fast(step, traj_infos, opt_info, test)
        start = time()
        with (self.experiment.test() if test else nullcontext()):
            step *= self.batch_size
            if opt_info is not None:
                # grad norm is left on the GPU for some reason
                # https://github.com/astooke/rlpyt/issues/163
                self.experiment.log_metrics(
                    {
                        k: np.mean(v)
                        for k, v in opt_info._asdict().items()
                        if k != "gradNorm"
                    },
                    step=step,
                )

            if traj_infos:
                agg_vals = {}
                for key in traj_infos[0].keys():
                    if key in self._cum_metrics:
                        continue
                    agg_vals[key] = sum(t[key] for t in traj_infos) / len(traj_infos)
                self.experiment.log_metrics(agg_vals, step=step)

            if not test:
                now = time()
                self.experiment.log_metrics(
                    {
                        "new_completed_trajs": self._new_completed_trajs,
                        "steps_per_second": (step - self._last_step)
                        / (now - self._last_time),
                    },
                    step=step,
                )
                self._last_time = now
                self._last_step = step
                self._new_completed_trajs = 0

        self.experiment.log_metrics(self._cum_metrics, step=step)
        self._cum_metrics["logging_time"] += time() - start

    def log_metric(self, name, val):
        self.experiment.log_metric(name, val)

    def log_parameters(self, parameters):
        self.experiment.log_parameters(parameters)

    def log_config(self, config):
        self.experiment.log_parameter("config", json.dumps(convert_dict(config)))

    def upload_snapshot(self):
        if self.snapshot_dir:
            self.experiment.log_asset(self._previous_snapshot_fname)

    def save_itr_params(
        self, step: int, params: Dict[str, Any], metric: Optional[float] = None
    ) -> None:
        super().save_itr_params(step, params, metric)
        now = time()
        if now - self._last_snapshot_upload > self._snaphot_upload_time:
            self._last_snapshot_upload = now
            self.upload_snapshot()

    def shutdown(self, error: bool = False) -> None:
        if not error:
            self.upload_snapshot()
            self.experiment.log_parameter("complete", True)
        self.experiment.end()


class PrintExperiment:
    def log_metric(self, name: str, val: float, step: Optional[int] = None) -> None:
        print(f"Metric {name}: {val}" + (f" ({step})" if step is not None else ""))

    def log_metrics(
        self, metrics: Dict[str, float], step: Optional[int] = None
    ) -> None:
        for name, val in metrics.items():
            self.log_metric(name, val, step)

    def log_parameter(self, name: str, val: Any, step: Optional[int] = None) -> None:
        print(f"Param {name}: {val}" + (f" ({step}" if step is not None else ""))

    def log_parameters(
        self, params: Dict[str, float], step: Optional[int] = None
    ) -> None:
        for name, val in params.items():
            self.log_parameter(name, val, step)

    def test(self):
        return nullcontext()

    def log_asset(self, asset):
        if isinstance(asset, str):
            print(f"Asset {asset}.")
        else:
            print(f"Asset of size {len(asset)}.")

    def end(self):
        pass


def convert_dict(d):
    d_new = {}
    for k, v in d.items():
        if isinstance(v, dict):
            d_new[k] = convert_dict(v)
        else:
            try:
                json.dumps(v)
                d_new[k] = v
            except TypeError:
                d_new[k] = v.__class__.__name__
    return d_new
