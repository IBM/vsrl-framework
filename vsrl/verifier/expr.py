#
# Copyright (C) 2020 IBM. All Rights Reserved.
#
# See LICENSE.txt file in the root directory
# of this source tree for licensing information.
#

from __future__ import annotations

import copy
import logging
from functools import wraps
from inspect import signature
from typing import Any, Callable, List, Set, TypeVar, Union, cast

import numpy as np

F = TypeVar("F", bound=Callable[..., Any])

# the type technically won't be correct here; if you have __init__(self, x: Term)
# then wrapper should be (self, x: Union[int, float, Term]). I don't think such
# type substitution is supported by mypy, though; just ignore any type errors
# from expecting __init__ methods to work with ints/floats.
def maybe_convert_terms(func: F) -> F:
    """
    Wrap a function to convert int/float arguments to Numbers.

    This is useful so that we can write something like:

    ```python
    x, y = map(Variable, "xy")
    expr = 2 * x + y ** 3
    ```

    instead of needing to do

    ```python
    x, y = map(Variable, "xy")
    expr = Number(2) * x + y ** Number(3)
    ```
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        args = [Number(arg) if isinstance(arg, (int, float)) else arg for arg in args]
        kwargs = {
            name: Number(arg) if isinstance(arg, (int, float)) else arg
            for name, arg in kwargs.items()
        }
        return func(*args, **kwargs)

    wrapper = cast(F, wrapper)
    return wrapper


class Node:
    # TODO explain the object protocols for setting and using AST state
    def children(self) -> list:
        children = self._compute_children()
        assert isinstance(children, list), "found a non-list: %s for %s" % (
            children,
            self.pretty_string(),
        )
        return children

    def clear_state(self):
        if hasattr(self, "state"):
            del self.state
        for c in self.children():
            assert isinstance(c, Node)
            c.clear_state()

    def pretty_string(self):
        raise Exception("Deprecated. Call the specific printing method instead.")

    def _compute_children(self):  # subclasses w/ children should override this
        return MatchError()


class Expression(Node):
    # TODO define a custom hashing function that maintains adt equalities.
    def __hash__(self):
        return super.__hash__(self)

    def copy(self):
        if hasattr(self, "state"):
            logging.log(
                logging.WARN,
                "Always reset state before copying a tree. In: %s"
                % self.pretty_string(),
            )
        return copy.deepcopy(self)

    def pretty_string(self):
        return pp(self)

    def __str__(self):
        return pp(self)  # todo make this a configuration option.

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        return self.is_referentially_eq(other)

    def is_referentially_eq(self, other):
        return hash(self) == hash(other)


class Program(Expression):
    pass


class Formula(Expression):
    def _compute_children(self):
        if hasattr(self, "left"):
            return [self.left, self.right]
        elif hasattr(self, "child"):
            return [self.child]
        elif isinstance(self, Box):
            return [self.program, self.formula]
        elif isinstance(self, Diamond):
            return [self.program, self.formula]
        elif isinstance(self, CompositeExpression):
            raise MatchError("Expected all composite expressions to have children.")
        else:
            return []


class Term(Expression):
    def _compute_children(self):
        if isinstance(self, CompositeExpression):
            if hasattr(self, "left"):
                return [self.left, self.right]
            elif hasattr(self, "child"):
                return [self.child]
            elif isinstance(self, Power):
                return [self.base, self.exp]
            else:
                raise MatchError(
                    "Expected all composite expressions to have children but didn't find any for %s"
                    % self.pretty_string()
                )
        else:
            return []

    #
    # def __add__(self, other):
    #     return Plus(self, other)
    #
    # def __mul__(self, other):
    #     return Times(self, other)
    #
    # def __pow__(self, other):
    #     return Power(self, other)
    #
    # def __neg__(self):
    #     return Neg(self)
    #
    # def __sub__(self, other):
    #     return Minus(self, other)
    #
    # def __ne__(self, other):
    #     return NotEq(self, other)


class CompositeExpression(Node):
    def arity(self):
        return len(signature(self.__init__).parameters.keys())


############################################################
# region Terms
############################################################


class Variable(Term):
    def __init__(self, name: str):
        self.name = name

    def __lt__(self, other: Variable) -> bool:
        """Define a less than operator so that a list of variables has a unique order."""
        assert isinstance(other, Variable)
        return self.name < other.name

    def __hash__(self):
        return hash("v" + self.name)


class Number(Term):
    def __init__(self, val: Union[float, int, np.int, np.int64, np.float, np.float64]):
        # todo use a bignum library.
        valid_types = (float, int, np.int, np.int64, np.float, np.float32, np.float64)
        if not isinstance(val, valid_types):
            raise ValueError(
                f"Expected a float or int as input but found: {val} (type = {type(val)})"
            )
        self.val = val

    def __hash__(self):
        return hash(self.val)


class Plus(Term, CompositeExpression):
    @maybe_convert_terms
    def __init__(self, left: Term, right: Term):
        self.left, self.right = left, right


class Times(Term, CompositeExpression):
    @maybe_convert_terms
    def __init__(self, left: Term, right: Term):
        self.left, self.right = left, right


class Minus(Term, CompositeExpression):
    @maybe_convert_terms
    def __init__(self, left: Term, right: Term):
        self.left, self.right = left, right


class Divide(Term, CompositeExpression):
    @maybe_convert_terms
    def __init__(self, left: Term, right: Term):
        self.left, self.right = left, right


class Power(Term, CompositeExpression):
    @maybe_convert_terms
    def __init__(self, base: Term, exp: Term):
        self.base, self.exp = base, exp


class Neg(Term, CompositeExpression):
    def __init__(self, child: Term):
        self.child = child


class Function(Expression):
    def __init__(self, name):
        self.name = name


class FuncApp(Term):
    def __init__(self, f: Function, child: Term):
        self.f = f
        self.child = child


class DotTerm(Term):
    def __init__(self, number: int = None, is_numeric=False, is_variable=False):
        assert not (is_numeric and is_variable)
        self.number = number
        self.is_numeric = is_numeric
        self.is_variable = is_variable


# endregion

############################################################
# region Formulas
############################################################


class And(Formula, CompositeExpression):
    def __init__(self, left: Formula, right: Formula):
        (self.left, self.right) = (left, right)


class Or(Formula, CompositeExpression):
    def __init__(self, left: Formula, right: Formula):
        (self.left, self.right) = (left, right)


class Not(Formula, CompositeExpression):
    def __init__(self, child: Formula):
        self.child = child


class Imply(Formula, CompositeExpression):
    def __init__(self, left: Formula, right: Formula):
        assert isinstance(left, Formula) and isinstance(right, Formula)
        (self.left, self.right) = (left, right)


class Equiv(Formula, CompositeExpression):
    def __init__(self, left: Formula, right: Formula):
        (self.left, self.right) = (left, right)


class Forall(Formula, CompositeExpression):
    def __init__(self, variables: Set[Variable], child: Formula):
        self.variables, self.child = variables, child


class Exists(Formula, CompositeExpression):
    def __init__(self, variables: Set[Variable], child: Formula):
        self.variables, self.child = variables, child


class Box(Formula, CompositeExpression):
    def __init__(self, program: Program, formula: Formula):
        assert isinstance(program, Program), f"expected program but found {program}"
        assert isinstance(formula, Formula), f"expected formula but found {formula}"
        (self.program, self.formula) = (program, formula)


class Diamond(Formula, CompositeExpression):
    def __init__(self, program: Program, formula: Formula):
        (self.program, self.formula) = (program, formula)


class Bool(Formula):
    pass


class TrueF(Bool):
    def __bool__(self):
        return True


class FalseF(Bool):
    def __bool__(self):
        return False


class Greater(Formula, CompositeExpression):
    def __init__(self, left: Term, right: Term):
        self.left, self.right = left, right


class Less(Formula, CompositeExpression):
    def __init__(self, left: Term, right: Term):
        self.left, self.right = left, right


class GreaterEq(Formula, CompositeExpression):
    def __init__(self, left: Term, right: Term):
        self.left, self.right = left, right


class LessEq(Formula, CompositeExpression):
    def __init__(self, left: Term, right: Term):
        self.left, self.right = left, right


class NotEq(Formula, CompositeExpression):
    def __init__(self, left: Term, right: Term):
        self.left, self.right = left, right


class Eq(Formula, CompositeExpression):
    def __init__(self, left: Term, right: Term):
        self.left, self.right = left, right


class Predicate(Expression):
    def __init__(self, name: str):
        self.name = name


class PredApp(Formula, CompositeExpression):
    def __init__(self, pred: Predicate, args: List[Expression]):
        self.pred, self.args = (pred, args)


class DotFormula(Formula):
    def __init__(self, number: int = None):
        self.number = number


# endregion

############################################################
# region Programs
############################################################


class Assign(Program, CompositeExpression):
    def __init__(self, v: Variable, e: Term):
        self.v = v
        self.e = e


class Test(Program, CompositeExpression):
    def __init__(self, formula: Formula):
        self.formula = formula

    def _compute_children(self):
        return [self.formula]


class SequentialCompose(Program, CompositeExpression):
    def __init__(self, left: Program, right: Program):
        self.left = left
        self.right = right


class Loop(Program, CompositeExpression):
    def __init__(self, child: Program):
        self.child = child


class Choice(Program, CompositeExpression):
    def __init__(self, left, right):
        self.left, self.right = left, right


class DifferentialProgram(Node):
    def copy(self):
        return copy.deepcopy(self)


class ODESystem(Program, CompositeExpression):
    def __init__(self, dp: DifferentialProgram, constraint: Formula):
        assert isinstance(dp, DifferentialProgram)
        assert isinstance(constraint, Formula)
        self.dp = dp
        self.constraint = constraint

    def _compute_children(self):
        return [self.dp, self.constraint]


class AtomicODE(DifferentialProgram, CompositeExpression):
    def __init__(self, x: Variable, e: Term):
        self.x = x
        self.e = e

    def _compute_children(self):
        return [self.x, self.e]


class DiffPair(DifferentialProgram, CompositeExpression):
    def __init__(self, left: DifferentialProgram, right: DifferentialProgram):
        assert isinstance(left, DifferentialProgram) and isinstance(
            right, DifferentialProgram
        )
        (self.left, self.right) = (left, right)
        self._auto_right_assoc()

    def _compute_children(self):
        return [self.left, self.right]

    def _auto_right_assoc(self) -> DifferentialProgram:
        """Re-associates the LHS and RHS so that the ODE as the form (a, (b, (c, ...)))

        :param left:
        :param right:
        :return: The right-associated ODE.
        """
        while isinstance(self.left, DiffPair):
            old_left = self.left
            self.left = self.left.left
            # Note: this is a sneaky recursive call!
            self.right = DiffPair(old_left.right, self.right)

        assert not isinstance(self.left, DiffPair), "Left should now be a non-pair!"


class DiffConst(DifferentialProgram):
    def __init__(self, c: str):
        self.c = c

    def _compute_children(self):
        return []


# endregion


# region Exceptions related to pattern matching style code for ASTs.


class MatchError(Exception):
    def __init__(self, msg=None):
        self.msg = msg

    def __str__(self):
        return (
            "Non-exhaustive pattern match against an algebraic data-type. Client-provided details: %s"
            % self.msg
        )


# endregion


def pp(e: Expression) -> str:
    """Pretty-prints differential dynamic logic expressions.

    :param e: The expression to print.
    :return: the pretty-print of the expression.
    """
    if isinstance(e, Program):
        return program_pp(e)
    elif isinstance(e, Formula):
        return formula_pp(e)
    elif isinstance(e, Term):
        return term_pp(e)
    elif isinstance(e, DifferentialProgram):
        return ode_pp(e)
    else:
        raise MatchError("Missing case: %s" % e.__class__)


# region Helper functions


def parens(ast_node: Expression, parent_node=None) -> str:
    """Pretty-prints an expression with any necessary parentheses.

    Prints the appropriate type of parentheses if necessary by considering the node being printed and its parent node.

    TODO currently adds parentheses no matter what.

    :param ast_node: the node being printed.
    :param parent_node: the parent of the node being printed.
    :return: the pretty-printed expression with parentheses if necessary from context.
    """
    if (
        (isinstance(ast_node, (Number, Variable, DifferentialProgram)))  # DEs commute
        or (isinstance(ast_node, Times) and isinstance(parent_node, (Plus, Times, Neg)))
        or (isinstance(ast_node, Neg) and isinstance(parent_node, Plus))
    ):
        return pp(ast_node)
    elif isinstance(ast_node, (Term, Formula)):
        return "(" + pp(ast_node) + ")"
    elif isinstance(ast_node, Program):
        return "{" + pp(ast_node) + "}"
    else:
        raise MatchError(
            f"Do not know how to add parens to .{ast_node}. of type {ast_node.__class__}"
        )


# endregion


def term_pp(e: Term) -> str:
    if isinstance(e, Number):
        return "%.2f" % e.val
    elif isinstance(e, Variable):
        return e.name
    elif isinstance(e, Plus):
        return parens(e.left, e) + "+" + parens(e.right, e)
    elif isinstance(e, Times):
        return parens(e.left, e) + "*" + parens(e.right, e)
    elif isinstance(e, Minus):
        return parens(e.left, e) + "-" + parens(e.right, e)
    elif isinstance(e, Divide):
        return parens(e.left, e) + "/" + parens(e.right, e)
    elif isinstance(e, Power):
        return parens(e.base, e) + "^" + parens(e.exp, e)
    elif isinstance(e, Neg):
        return "-" + parens(e.child, e)
    elif isinstance(e, DotTerm):
        idx = "_%d" % e.number if e.number is not None else ""
        decorator = (
            "@Number" if e.is_numeric else ("@Variable" if e.is_variable else "")
        )
        return "•%s%s" % (idx, decorator)
    elif isinstance(e, FuncApp):
        return e.f.name + parens(e.child, e)
    else:
        raise MatchError(f"Do not know how to parse term of class ${e.__class__}")


def formula_pp(f: Formula) -> str:
    if isinstance(f, TrueF):
        return "true"
    elif isinstance(f, FalseF):
        return "false"
    elif isinstance(f, And):
        return parens(f.left, f) + " & " + parens(f.right, f)
    elif isinstance(f, Or):
        return parens(f.left, f) + " | " + parens(f.right, f)
    elif isinstance(f, Imply):
        return parens(f.left, f) + " -> " + parens(f.right, f)
    elif isinstance(f, Not):
        return "!" + parens(f.child, f)
    elif isinstance(f, Box) or isinstance(f, Diamond):
        return modality_pp(f)
    elif isinstance(f, Forall):
        rv = ""
        for v in f.variables:
            rv = rv + "\\forall" + v.name + "."
        rv = rv + parens(f.child, f)
        return rv
    elif isinstance(f, Exists):
        rv = ""
        for v in f.variables:
            rv = rv + "\\exists" + v.name + "."
        rv = rv + parens(f.child, f)
        return rv
    elif isinstance(f, Greater):
        return parens(f.left, f) + " > " + parens(f.right, f)
    elif isinstance(f, Less):
        return parens(f.left, f) + " < " + parens(f.right, f)
    elif isinstance(f, GreaterEq):
        return parens(f.left, f) + " >= " + parens(f.right, f)
    elif isinstance(f, LessEq):
        return parens(f.left, f) + " <= " + parens(f.right, f)
    elif isinstance(f, Eq):
        return parens(f.left, f) + " = " + parens(f.right, f)
    elif isinstance(f, NotEq):
        return parens(f.left, f) + " != " + parens(f.right, f)
    elif isinstance(f, DotFormula):
        return "•_%d" % f.number if f.number is not None else "•"
    else:
        raise MatchError(f"Do not know how to pretty-print ${f.__class__}")


def program_pp(e: Program) -> str:
    if isinstance(e, ODESystem):
        return "{" + ode_pp(e.dp) + " & " + formula_pp(e.constraint) + "}"
    elif isinstance(e, Choice):
        return "{ {" + program_pp(e.left) + "} ++ {" + program_pp(e.right) + "} }"
    elif isinstance(e, SequentialCompose):
        return "{ {" + program_pp(e.left) + "} ; {" + program_pp(e.right) + "} }"
    elif isinstance(e, Loop):
        return "{" + program_pp(e.child) + "}*"
    elif isinstance(e, Test):
        return "?" + formula_pp(e.formula)
    elif isinstance(e, Assign):
        return term_pp(e.v) + " := " + term_pp(e.e)
    else:
        raise MatchError(f"Not sure what to do with program ${e.__class__}")


def ode_pp(e: DifferentialProgram) -> str:
    if isinstance(e, AtomicODE):
        return "%s' = %s" % (e.x.name, term_pp(e.e))
    elif isinstance(e, DiffPair):
        # note: diff pairs are left-associative, so we could also do a fold left to avoid recursion
        # if the pretty printer becomes a bottle neck.
        return ode_pp(e.left) + "," + ode_pp(e.right)
    elif isinstance(e, DiffConst):
        return e.c
    else:
        raise MatchError(
            f"Expected and atomicode, diffpair, or diffconst but fond: {e.__class__}"
        )


def modality_pp(f: Formula) -> str:
    if isinstance(f, Box):
        return "[" + program_pp(f.program) + "]" + formula_pp(f.formula)
    elif isinstance(f, Diamond):
        return "<" + program_pp(f.program) + ">" + formula_pp(f.formula)
    else:
        raise MatchError()
