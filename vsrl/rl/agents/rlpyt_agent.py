#
# Copyright (C) 2020 IBM. All Rights Reserved.
#
# See LICENSE.txt file in the root directory
# of this source tree for licensing information.
#

import random
from typing import Any

import numpy as np
import rlpyt.agents.base
import torch.Tensor

from vsrl.rl.agents.agent import Agent, Monitor
from vsrl.rl.envs import Env
from vsrl.spaces.space import Space
from vsrl.symmap.symbolic_mapper import SymbolicMapper


class RlpytConstrainedAgent(Agent):
    """
    Wraps a `rlpyt.agents.base.BaseAgent`
    TODO The agent performs the symbolic mapping and passes back to the environment
         an agentinfo containing the symbolic features, but does NOT perform the safety checking.
    """

    def __init__(
        self,
        rlpyt_agent: rlpyt.agents.base.BaseAgent,
        m: Monitor,
        sm: SymbolicMapper,
        action_space: Space,
    ) -> None:
        self._monitor: Monitor = m
        self._symbolic_mapper: SymbolicMapper = sm
        self.action_space = action_space
        self.rlpyt_agent = rlpyt_agent

        self._prev_action = None
        self._prev_reward = None

    def act(self, obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "Need to call act with prev aciton and prev reward when interfacing with rlpyt."
        )

    def step(self, obs: np.ndarray, prev_state, prev_action) -> np.ndarray:
        # extract symbolic feature.
        symbolic_state = self._symbolic_mapper(obs)

        # get the agent's action.
        agent_step: rlpyt.agents.base.AgentStep = self.rlpyt_agent.step(
            obs, prev_state, prev_action
        )
        actions: torch.Tensor = agent_step.action
        # todo maybe we have to squeeze here? not sure.
        actions_as_numpy = actions.detach().cpu().numpy()
        assert (
            actions_as_numpy.shape[0] == 1
        ), "Do not currently support acting in multiple environments. TODO parallelize."
        action: np.ndarray = actions_as_numpy[0]
        assert action in self.action_space

        # choose a safe action.
        safe_action = None
        if self.monitor.ctrl_action_is_safe(
            symbolic_state, self.action_space.to_state(action)
        ):
            safe_action = action
        else:
            safe_action = self.action_space.constrained_sample(
                self._monitor.controller_monitor_in_state(symbolic_state)
            )
        # todo now actually take the step.

    def end(self):
        pass

    @property
    def monitor(self) -> Monitor:
        return self._monitor
