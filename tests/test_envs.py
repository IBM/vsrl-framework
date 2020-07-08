#
# Copyright (C) 2020 IBM. All Rights Reserved.
#
# See LICENSE.txt file in the root directory
# of this source tree for licensing information.
#

import pytest

from vsrl.rl.envs import ACC, PMGoalFinding, Pointmesses

# from vsrl.rl.rlpyt.env import RLPytEnv

ENVS = [ACC, PMGoalFinding, Pointmesses]
N_STEPS = 1_000


@pytest.mark.parametrize("Env", ENVS)
def test_step_random(Env):
    env = Env()
    env.reset()
    for _ in range(N_STEPS):
        action = env.action_space.sample()
        _, _, done, _ = env.step(action)
        if done:
            env.reset()


# TODO: finish this
# @pytest.mark.parametrize("Env", ENVS)
# def test_oracle_safety_random(Env):
#     env = RLPytEnv(Env())
#     env.reset()
#     for _ in range(N_STEPS):
#         action = ...
#         _, _, done, info = env.step(action, sym_features=env._env._state)
#         assert not info.unsafe
#         if done:
#             env.reset()
