#
# Copyright (C) 2020 IBM. All Rights Reserved.
#
# See LICENSE.txt file in the root directory
# of this source tree for licensing information.
#

"""
Tests:
* training a detector for 1 epoch
* loading in a detector from the saved checkpoint file
* using the detector on actual observations from an environment
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from vsrl.rl.envs import ACC, PMGoalFinding, Pointmesses
from vsrl.training import train_center_track

# from vsrl.symmap.detectors import CenterTrack

ENVS = [ACC, PMGoalFinding, Pointmesses]


def get_config_for_env(env: str) -> Path:
    return Path(__file__).parents[1].resolve() / "assets" / "configs" / f"{env}.toml"


@pytest.mark.parametrize("Env", ENVS)
def test_detector_no_error(Env) -> None:
    config_path = get_config_for_env(Env.__name__)
    with TemporaryDirectory(prefix="models") as save_dir:
        model = train_center_track(
            config_path,
            save_dir,
            max_epochs=1,
            epoch_size=100,
            img_scale=4,
            grayscale=True,
            use_logger=False,
            prog_bar=False,
        )
    # TODO: load detector from checkpoint, ensure it matches the current model


# TODO: finish this
# @pytest.mark.parametrize("Env", ENVS)
# def test_(Env) -> None:
#     config_path = get_config_for_env(Env.__name__)
#     model = CenterTrack("resnet_small", config_path)
#     env = Env()
#     env.reset()
#     for _ in range(100):
#         action = env.action_space.sample()
#         obs, *_ = env.step(action)
# test model on obs (just to see it works; not checking against ground-truth)
