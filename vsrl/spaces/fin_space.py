#
# Copyright (C) 2020 IBM. All Rights Reserved.
#
# See LICENSE.txt file in the root directory
# of this source tree for licensing information.
#

import random
from typing import Dict, List

import numpy as np

from vsrl.verifier import expr

from .space import Space


class FiniteSpace(Space):
    def __init__(self, elements: List[np.ndarray], dim_names: List[expr.Variable]):
        assert len(elements) > 0
        assert isinstance(elements[0], np.ndarray)
        self._elements = elements
        self._dim_names = dim_names

        assert self._check_elements_contract()
        assert self._check_dim_names_contract()

    def _check_elements_contract(self):
        """Conditions on the well-formedness of elements."""
        assert self._elements, "Expected some elements."
        for e in self._elements:
            assert isinstance(e, np.ndarray)
        return True

    def _check_dim_names_contract(self):
        assert isinstance(self.dimension_names, list)
        assert (
            len(self.dimension_names) == self.dimension()
        ), f"{len(self.dimension_names)} != {self.dimension()}"
        assert len(self.dimension_names) > 0
        for name in self.dimension_names:
            assert isinstance(name, expr.Variable)
        return True

    def dimension(self):
        assert len(self.dimension_names) == len(self.elements[0])
        return len(self.dimension_names)

    def index_of_element(self, element):
        assert element in self.elements
        for i, e in enumerate(self.elements):
            if element == e:
                return i
        raise Exception(
            "element \in elements but couldn't find it by enumerating. This should not happen."
        )

    def __contains__(self, item):
        return item in self.elements

    def is_finite(self):
        return True

    def __iter__(self):
        return FiniteSpaceIterator(self)

    @property
    def elements(self):
        return self._elements

    @property
    def dimension_names(self) -> List[expr.Variable]:
        return self._dim_names

    def sample(self):
        rv = random.choice(self.elements)
        return rv

    def constrained_sample(self, constraint: expr.Formula) -> np.ndarray:
        raise NotImplementedError

    def _values_are_scalars(self, d: Dict):
        for k in d.keys():
            assert isinstance(k, expr.Expression), k
            assert (
                isinstance(d[k], float)
                or isinstance(d[k], int)
                or isinstance(d[k], np.ndarray)
            ), d[k]
        return True

    def constrained_max(
        self, f: expr.Term, constraint: expr.Formula, state: Dict[expr.Variable, object]
    ) -> np.ndarray:
        raise NotImplementedError


class FiniteSpaceIterator:
    def __init__(self, space):
        self._space: FiniteSpace = space
        self._idx = 0

    def __next__(self):
        if self._idx < len(self._space.elements):
            result = self._space.elements[self._idx]
            self._idx += 1
            return result
        else:
            raise StopIteration
