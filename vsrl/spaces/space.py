#
# Copyright (C) 2020 IBM. All Rights Reserved.
#
# See LICENSE.txt file in the root directory
# of this source tree for licensing information.
#

import random
import sys
from abc import ABC, abstractmethod
from typing import Dict, List, Union

import numpy as np

import vsrl.verifier.expr as expr


class Space(ABC):
    """
    A Space is a set of things that you can sample from.
    IBM VSRL Spaces are similar to OpenAI Gym spaces, with two important differences:

    1. VSRL Spaces implement methods for constrained sampling.
    2. Each element of a VSRL Space corresponds to a State.

    States are simply mappings from expr.Variables to expr.Terms (Dict[expr.Variable, expr.Term]).
    The identification of Space elements with states is done so that the input to constrained_sample can be a formula
    whose free variables are given by the state.
    """

    @abstractmethod
    def __contains__(self, element: np.ndarray) -> bool:
        assert hasattr(
            element, "__iter__"
        ), "Elements should always be vectors or ndarrays"
        raise NotImplementedError

    def __iter__(self):
        if self.is_finite():
            raise NotImplementedError
        else:
            raise Exception("Cannot iterate over an infinite set.")

    @abstractmethod
    def is_finite(self) -> bool:
        """Finite spaces should implement an iterator."""
        raise NotImplementedError

    @abstractmethod
    def dimension(self) -> int:
        """ Returns the dimension of the space. """

    @property
    @abstractmethod
    def dimension_names(self) -> List[expr.Variable]:
        """ Returns the names of each dimension. This should always be the keyset used in from_state, and these variables
            should be the variable names returned in the formulas passed into constrained_sample """

    def from_state(self, state: Dict[expr.Variable, expr.Expression]) -> np.ndarray:
        assert isinstance(state, dict)
        es = state.values()
        array = np.ndarray(len(state.keys()))  # todo support more complex shapes.
        for i, e in enumerate(es):
            assert isinstance(e, expr.Number)
            array[i] = e.val
        assert array in self
        return array

    def _expr_to_ndarray_element(self, e: expr.Expression):
        if isinstance(e, expr.Number):
            return e.val

    def to_state(self, element: np.ndarray) -> Dict[expr.Variable, expr.Expression]:
        # warning: do not assert that element is in self; this can result in infinite recursion.
        assert isinstance(element, np.ndarray)
        assert hasattr(
            element, "__iter__"
        ), "Elements should always be vectors or ndarrays"
        return dict(zip(self.dimension_names, map(self._to_term, element)))

    def _to_term(self, x: Union[float, int, np.float64]) -> expr.Number:
        if (
            isinstance(x, float)
            or isinstance(x, np.float)
            or isinstance(x, np.float64)
            or isinstance(x, np.float32)
            or isinstance(x, np.int)
            or isinstance(x, np.int64)
        ):
            return expr.Number(x)
        else:
            raise NotImplementedError(
                f"Conversion to term not implemented for {x} of type {x.__class__}"
            )

    def null_value(self):
        """the canonical witness that the space is non-empty."""

    @abstractmethod
    def sample(self) -> np.ndarray:
        """ Samples from the space. """

    @abstractmethod
    def constrained_sample(self, constraint: expr.Formula) -> np.ndarray:
        """ Samples from the space subject to a constraint.
        :return: constraint-satisfying values for any variables that occur in free_vars(constraint) but not in state.keys()
        """

    @abstractmethod
    def constrained_max(
        self,
        f: expr.Term,  # todo maybe this needs to be an object not a Term.
        constraint: expr.Formula,
    ) -> np.ndarray:
        """ returns argmax_space(f) subject to the constraint.
            TODO not sure if this makes any sense...
        :param f: The function to optimize.
        :param constraint: The constraint
        :return: max constraint-satisfying values for any variables that occur in free_vars(constraint) but not in state.keys()
        """

    @property
    def float_range(self):
        return sys.float_info.min, sys.float_info.max

    def random_float(self):
        return random.uniform(self.float_range[0], self.float_range[1])
