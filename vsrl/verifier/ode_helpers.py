#
# Copyright (C) 2020 IBM. All Rights Reserved.
#
# See LICENSE.txt file in the root directory
# of this source tree for licensing information.
#

from typing import Dict, List, Set

from vsrl.verifier import expr

def atomic_odes(ode: expr.ODESystem) -> Dict[expr.Variable, expr.Term]:
    return _atomic_odes(ode.dp)


def _atomic_odes(ode: expr.DifferentialProgram) -> Dict[expr.Variable, expr.Term]:
    if isinstance(ode, expr.AtomicODE):
        return {ode.x: ode.e}
    elif isinstance(ode, expr.DiffPair):
        left_atomics = _atomic_odes(ode.left)
        assert left_atomics is not None
        left_atomics.update(_atomic_odes(ode.right))
        return left_atomics
    else:
        raise expr.MatchError(
            "differential constants have infinite primed variables, so we need lattices for them: %s"
            % ode
        )


def primed_vars(ode: expr.ODESystem) -> List[expr.Variable]:
    return _dp_primed_vars(ode.dp)


def _dp_primed_vars(ode: expr.DifferentialProgram) -> List[expr.Variable]:
    if isinstance(ode, expr.AtomicODE):
        s = list()
        s.append(ode.x)
        return s
    elif isinstance(ode, expr.DiffPair):
        left = _dp_primed_vars(ode.left)
        right = _dp_primed_vars(ode.right)
        left.extend(right)
        return left
    else:
        raise expr.MatchError(
            "differential constants have infinite primed variables, so we need lattices for them."
        )


def rhs_for_primed_var(ode: expr.ODESystem, v: expr.Variable) -> expr.Term:
    """:return: The right-hand side of the equation v'=e in the odes."""
    assert v in primed_vars(ode), (
        "cannote get the rhs for %s because it's not in the ODEs %s"
        % (v, ode.pretty_string())
    )

    return _rhs_for_primed_var(ode.dp, v)


def right_hand_sides(ode: expr.ODESystem) -> Set[expr.Formula]:
    return atomic_odes(ode).values()


def _rhs_for_primed_var(dp: expr.DifferentialProgram, v: expr.Variable) -> expr.Term:
    """:return: The right-hand side of the equation v'=e in the odes."""
    if isinstance(dp, expr.AtomicODE) and dp.x.is_referentially_eq(v):
        return dp.e
    elif isinstance(dp, expr.AtomicODE) and not dp.x.is_referentially_eq(v):
        return None
    elif isinstance(dp, expr.DiffPair):
        left_result = _rhs_for_primed_var(dp.left, v)
        right_result = _rhs_for_primed_var(dp.right, v)
        assert not (
            left_result is not None and right_result is not None
        ), "expected either left or right to contain v but not both"
        if left_result is not None:
            return left_result
        elif right_result is not None:
            return right_result
        else:
            return None
    else:
        raise expr.MatchError()
