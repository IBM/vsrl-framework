#
# Copyright (C) 2020 IBM. All Rights Reserved.
#
# See LICENSE.txt file in the root directory
# of this source tree for licensing information.
#

from abc import ABC, abstractmethod
from typing import Dict

import vsrl.verifier.expr_helpers
from vsrl.verifier import expr as expr


class Monitor(ABC):
    @property
    @abstractmethod
    def controller_monitor(self) -> expr.Formula:
        raise NotImplementedError

    @property
    @abstractmethod
    def model_monitor(self) -> expr.Formula:
        raise NotImplementedError

    @abstractmethod
    def ctrl_action_is_safe(
        self,
        state: Dict[expr.Variable, expr.Number],
        action: Dict[expr.Variable, expr.Number],
    ) -> bool:
        raise NotImplementedError("A monitor should implement a controller_monitor.")

    def controller_monitor_in_state(self, state: Dict[expr.Variable, expr.Number]):
        """
        The is the predicate `f(a)` such that:
        {{{
            f(a) <-> self.ctrl_action_is_safe(state, a)
        }}}

        :param state: The state in which the controller monitor is being evaluated.
        :return: A formula whose only free variables are the variables in the action space.
        """
        return vsrl.verifier.expr_helpers.multi_replace(state, self.controller_monitor)

    @abstractmethod
    def model_is_accurate(
        self,
        state: Dict[expr.Variable, expr.Number],
        action: Dict[expr.Variable, expr.Number],
        next_state: Dict[expr.Variable, expr.Number],
    ) -> bool:
        raise NotImplementedError(
            "A monitor should optionally implement a model_monitor."
        )
