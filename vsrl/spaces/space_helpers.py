#
# Copyright (C) 2020 IBM. All Rights Reserved.
#
# See LICENSE.txt file in the root directory
# of this source tree for licensing information.
#

"""
Helper methods for spaces and states.
"""

import copy
from typing import Dict

import vsrl.verifier.expr as expr


def union_states(
    s1: Dict[expr.Variable, expr.Expression], s2: Dict[expr.Variable, expr.Expression]
) -> Dict[expr.Variable, expr.Expression]:
    """Returns the union of 2 disjoint states."""
    assert s1.keys().isdisjoint(s2.keys()), "States should be disjoint."
    return_value = copy.deepcopy(s1)
    return_value.update(s2)
    return return_value


def assert_is_state(s: Dict[expr.Variable, expr.Expression]) -> bool:
    assert isinstance(s, dict)
    assert len(s.keys()) > 0
    for k in s.keys():
        v = s[k]
        assert isinstance(k, expr.Variable)
        assert isinstance(v, expr.Expression)
    return True


def is_state(s: Dict[expr.Variable, expr.Expression]) -> bool:
    try:
        return assert_is_state(s)
    except AssertionError:
        return False
