#
# Copyright (C) 2020 IBM. All Rights Reserved.
#
# See LICENSE.txt file in the root directory
# of this source tree for licensing information.
#

from vsrl.verifier.expr import *

# region implementation of on_every_node


def on_every_node(fun, e: Expression):
    if isinstance(e, Formula):
        _on_every_formula(fun, e)
    elif isinstance(e, Program):
        _on_every_program(fun, e)
    elif isinstance(e, Term):
        _on_every_term(fun, e)
    elif isinstance(e, DifferentialProgram):
        _on_every_dp(fun, e)


def _on_every_program(fun, p: Program):
    fun(p)
    if isinstance(p, ODESystem):
        _on_every_dp(fun, p.dp)
        _on_every_formula(fun, p.constraint)
    else:
        raise NotImplementedError()


def _on_every_term(fun, e: Term):
    fun(e)
    if isinstance(e, Number):
        pass
    elif isinstance(e, Variable):
        pass
    elif isinstance(e, Plus):
        _on_every_term(fun, e.left)
        _on_every_term(fun, e.right)
    elif isinstance(e, Times):
        _on_every_term(fun, e.left)
        _on_every_term(fun, e.right)
    elif isinstance(e, Minus):
        _on_every_term(fun, e.left)
        _on_every_term(fun, e.right)
    elif isinstance(e, Divide):
        _on_every_term(fun, e.left)
        _on_every_term(fun, e.right)
    elif isinstance(e, Power):
        _on_every_term(fun, e.base)
        _on_every_term(fun, e.exp)
    elif isinstance(e, Neg):
        _on_every_term(fun, e.child)
    elif isinstance(e, DotTerm):
        pass
    else:
        raise MatchError(e.__class__)


def _on_every_dp(fun, dp: DifferentialProgram):
    fun(dp)
    if isinstance(dp, DiffPair):
        _on_every_dp(fun, dp.left)
        _on_every_dp(fun, dp.right)
    elif isinstance(dp, AtomicODE):
        _on_every_term(fun, dp.x)
        _on_every_term(fun, dp.e)
    elif isinstance(dp, DiffConst):
        pass
    else:
        raise MatchError(dp)


def _on_every_formula(
    fun, f: Formula, descend_into_terms=True, descend_into_programs=True
):
    fun(f)
    if isinstance(f, TrueF):
        pass
    elif isinstance(f, FalseF):
        pass
    elif isinstance(f, And):
        _on_every_formula(fun, f.left)
        _on_every_formula(fun, f.right)
    elif isinstance(f, Or):
        _on_every_formula(fun, f.left)
        _on_every_formula(fun, f.right)
    elif isinstance(f, Imply):
        _on_every_formula(fun, f.left)
        _on_every_formula(fun, f.right)
    elif isinstance(f, Not):
        _on_every_formula(fun, f.child)
    elif isinstance(f, Box) or isinstance(f, Diamond):
        if descend_into_programs:
            _on_every_program(fun, f.program)
        _on_every_formula(fun, f.formula)
    elif isinstance(f, Forall):
        if descend_into_terms:
            map(lambda v: _on_every_term(fun, v), f.variables)
        _on_every_formula(fun, f.child)
    elif isinstance(f, Exists):
        if descend_into_terms:
            map(lambda v: _on_every_term(fun, v), f.variables)
        _on_every_formula(fun, f.child)
    elif isinstance(f, Greater):
        if descend_into_terms:
            _on_every_term(fun, f.left)
            _on_every_term(fun, f.right)
    elif isinstance(f, Less):
        if descend_into_terms:
            _on_every_term(fun, f.left)
            _on_every_term(fun, f.right)
    elif isinstance(f, GreaterEq):
        if descend_into_terms:
            _on_every_term(fun, f.left)
            _on_every_term(fun, f.right)
    elif isinstance(f, LessEq):
        if descend_into_terms:
            _on_every_term(fun, f.left)
            _on_every_term(fun, f.right)
    elif isinstance(f, Eq):
        if descend_into_terms:
            _on_every_term(fun, f.left)
            _on_every_term(fun, f.right)
    elif isinstance(f, DotFormula):
        pass
    else:
        raise MatchError("%s %s" % (f.pretty_string(), f))


# endregion
