#
# Copyright (C) 2020 IBM. All Rights Reserved.
#
# See LICENSE.txt file in the root directory
# of this source tree for licensing information.
#

import logging
from typing import Dict, List, Tuple

import numpy as np

import vsrl.verifier.expr as vexpr

from .space import Space


class CompactSet(Space):
    """
    A compact real-valued set:
    `rLower_0 <= x_0 <= rUpper_0 & ...` where rLower_i, rUpper_i are real numbers.
    A special case of SAS in which each variable has a simple upper and lower bound.
    """

    def __init__(self, bounds: Dict[vexpr.Variable, Tuple[float, float]]):
        """
        :param bounds: {var: (lower_bound, upper_bound)}
        """
        self._names = list(bounds.keys())
        self.bounds: Dict[vexpr.Variable, (float, float)] = bounds
        self.lower_bounds = np.array(
            [lb for lb, _ in bounds.values()], dtype=np.float32
        )
        self.upper_bounds = np.array(
            [ub for _, ub in bounds.values()], dtype=np.float32
        )
        assert self._wfb()

    def _wfb(self):
        """ Checks that the bounds are well-formed. """
        assert isinstance(self.bounds, dict)
        for k in self.bounds.keys():
            assert isinstance(k, vexpr.Variable)
        for b in self.bounds.values():
            assert isinstance(
                b, tuple
            ), f"expected tuple but found {b} of type {b.__class__}"
            assert (
                len(b) == 2
            ), f"Expected a 2-tuple but found {b} in bounds {self.bounds}"
            assert isinstance(b[0], float) or isinstance(b[0], int)
            assert isinstance(b[1], float) or isinstance(b[1], int)
        return True

    def __contains__(self, element: np.ndarray) -> bool:
        assert isinstance(
            element, np.ndarray
        ), f"expected np.ndarray but found {element}"
        for i, e in enumerate(element):
            if element[i] < self.lower_bounds[i]:
                name = list(self.to_state(element).keys())[i]
                logging.info(
                    f"{name} element is too small because {element[i]} < {self.lower_bounds[i]}."
                )
                return False
            elif element[i] > self.upper_bounds[i]:
                name = list(self.to_state(element).keys())[i]
                logging.info(
                    f"{name} element is too big because {element[i]} > {self.upper_bounds[i]}."
                )
                return False
        return True
        # return (self.lower_bounds <= element).all() and (
        #     element <= self.upper_bounds
        # ).all()

    def is_finite(self) -> bool:
        return False

    def dimension(self) -> int:
        return len(self._names)

    @property
    def dimension_names(self) -> List[vexpr.Variable]:
        return self._names

    def sample(self) -> np.ndarray:
        return np.random.uniform(self.lower_bounds, self.upper_bounds)

    def constrained_sample(
        self, constraint: vexpr.Formula, state: Dict[vexpr.Variable, object]
    ) -> np.ndarray:
        # TODO create a Continuous Space and then create a constraint that includes all of the previous constrained
        # in addition to these bounds.
        raise NotImplementedError

    def constrained_max(
        self,
        space: Space,
        f: vexpr.Term,
        constraint: vexpr.Formula,
        state: Dict[vexpr.Variable, object],
    ) -> np.ndarray:
        raise NotImplementedError
