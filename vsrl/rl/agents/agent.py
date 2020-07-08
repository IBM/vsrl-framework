#
# Copyright (C) 2020 IBM. All Rights Reserved.
#
# See LICENSE.txt file in the root directory
# of this source tree for licensing information.
#

from abc import ABC, abstractmethod
from typing import Any, Optional

from vsrl.rl.envs import Env
from vsrl.verifier.monitor import Monitor


class Agent(ABC):
    @abstractmethod
    def act(self, obs: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def end(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def env(self) -> Optional[Env]:
        """The environment that this agent is acting in. Optional. """
        raise NotImplementedError


class ConstrainedAgent(Agent):
    @property
    @abstractmethod
    def monitor(self) -> Monitor:
        raise NotImplementedError
