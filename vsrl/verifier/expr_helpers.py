#
# Copyright (C) 2020 IBM. All Rights Reserved.
#
# See LICENSE.txt file in the root directory
# of this source tree for licensing information.
#

import copy
from typing import Dict, List, Optional, Set, Tuple

import vsrl.verifier.traversal as traversal
from vsrl.verifier.expr import *

# Set REPLACEMENT_LOGGING to True if you want detailed logging during replacement.
# This is not suggested unless debugging because running the pretty printer a few times costs about a half second.
# This is global rather than a parameter to the replacement function because oftentimes you need to see the entire
# trace of replacements in order to find an error. Changing all the client call sites to pass in a log flag to the
# replacement method would be annoying.
# TODO make this an option in the logging section of the vsrl config file instead of putting it here.
REPLACEMENT_LOGGING = False


def variables(e: Expression) -> Set[Variable]:
    """
    Returns all of the Variables in `e`.
    This also provides a simple example of how client code uses traversal.on_every_node.
    """
    return_value = set()

    def f(e: Expression):
        if isinstance(e, Variable):
            return_value.add(e)

    traversal.on_every_node(f, e)
    return return_value


def numbers(e: Expression) -> Set[Number]:
    """ Returns all of the numbers in `e`. """
    return_value = set()

    def f(e: Expression):
        if isinstance(e, Number):
            return_value.add(e)

    traversal.on_every_node(f, e)
    return return_value


def get_atomic_odes(ode_system: ODESystem) -> List[AtomicODE]:
    """ Returns all of the AtomicODEs in the `ode_system`. """
    atomic_odes = []

    def append_atomic_odes(expr):
        if isinstance(expr, AtomicODE):
            atomic_odes.append(expr)

    traversal.on_every_node(append_atomic_odes, ode_system.dp)
    return atomic_odes


def generate_fresh_variable(e: Expression) -> Variable:
    """ Returns a Variable that does not occur in `e`. """
    vars_in_e = []

    def _v_map(e: Expression):
        if isinstance(e, Variable):
            vars_in_e.append(e)

    traversal.on_every_node(_v_map, e)

    v = Variable("fv")
    while v in vars_in_e:
        v = Variable(v.name + "1")
    return v


def conjuncts(e: Expression) -> List[Formula]:
    """ Returns a list of conjuncts, assuming e a conjunction. Does not do any normalization. """
    if isinstance(e, And):
        left = conjuncts(e.left)
        right = conjuncts(e.right)
        return left.extend(right)
    else:
        return e


# region helpers for comparison formulas.


def is_comparison_operator(f: Formula):
    """ Returns true if f is a comparison formula (<, >, >=, <=, and =). """
    assert isinstance(f, Formula)
    return (
        isinstance(f, Greater)
        or isinstance(f, GreaterEq)
        or isinstance(f, Less)
        or isinstance(f, LessEq)
        or isinstance(f, Eq)
    )


def as_numeric_bound_on_variable(f: Formula) -> Optional[Tuple[Variable, Number]]:
    """
    If `f` has the form `v ~ r` or `r ~ v` where r is a number and ~ is a comparison operator, then this function will
    return (v,r). Otherwise, returns None.
    """
    assert isinstance(f, Formula)
    if not is_comparison_operator(f):
        return None
    else:
        if isinstance(f.left, Variable) and isinstance(f.right, Number):
            return f.left, f.right
        elif isinstance(f.left, Number) and isinstance(f.right, Variable):
            return f.rght, f.left
    return None


# endregion


def all_nonspecial_dots(e: Expression):
    """
    Returns all of the non-negative-indexed dots.

    We sometimes use negative-indexed dots when pipelining refinements in order to indicate what the next step in the
    pipeline should do. E.g., consider an algorithm that first traverses an expression to determine which terms should
    be constant but does not determine which constant to choose for each term. If it makes all of those "this should
    be a number" dots special (by setting their indices to negative values), then in then ext pass a sysid algorithm
    could be used to fill in concrete values for each indicated dot. This is just a convention; within the framework
    itself, there's nothing semantically special about negative indices on dots unless the client cod
    """
    return list(
        filter(
            lambda x: isinstance(x, DotFormula)
            or (not x.is_numeric and not x.is_variable),
            all_dots(e),
        )
    )


def all_dots(e: Expression):
    """
    Returns the set of all Dots in the expression `e`.
    A dot is a
    """
    dots = set()

    def _dot_map(e: Expression):
        if isinstance(e, DotTerm):
            dots.add(e)
        elif isinstance(e, DotFormula):
            dots.add(e)

    try:
        traversal.on_every_node(_dot_map, e)
    except MatchError:
        raise Exception(e)

    return list(dots)


def numeric_dots(e: Expression) -> List[DotTerm]:
    """ Returns the list of all DotTerms that must be filled by a number. """
    return list(
        filter(lambda dot: isinstance(dot, DotTerm) and dot.is_numeric, all_dots(e))
    )


def variable_dots(e: Expression) -> List[DotTerm]:
    """ Returns the list of all DotTerms that must be filled by a variable. """
    return list(
        filter(lambda dot: isinstance(dot, DotTerm) and dot.is_variable, all_dots(e))
    )


def true_term_dots(e: Expression) -> List[DotTerm]:
    """ Returns all of the term dots that are not variables or numbers. """
    return list(
        filter(
            lambda dot: isinstance(dot, DotTerm)
            and not dot.is_variable
            and not dot.is_numeric,
            all_dots(e),
        )
    )


def formula_dots(e: Expression) -> List[DotTerm]:
    """ Returns the set of all DotFormulas in `e`. """
    return list(filter(lambda dot: isinstance(dot, DotFormula), all_dots(e)))


def term_dots(e: Expression) -> List[DotTerm]:
    """ Returns the set of all DotTerms in `e`. """
    return list(filter(lambda dot: isinstance(dot, DotTerm), all_dots(e)))


def fresh_formula_dots(e: Expression, num_requested=1) -> List[DotFormula]:
    """ Generates `num_requested` FormulaDots that do not already occur in `e`. """
    dots = all_dots(e)
    new_dots = []
    i = 0
    while len(new_dots) != num_requested:
        while DotFormula(i) in dots:
            i = i + 1
        new_dots.append(DotFormula(i))
        i = i + 1
    assert len(new_dots) == num_requested
    return new_dots[0] if len(new_dots) == 1 else new_dots


def fresh_term_dots(
    e: Expression, num_requested=1, is_numeric=False, is_variable=False
) -> List[DotTerm]:
    """
    Generates `num_requested` TermDots that do not already occur in `e`.
    These does will be numbers if `if_numeric` or variables if `is_variable`.
    :param e: The expression.
    :param num_requested: The number of dots desired.
    :param is_numeric: True if the dot *must* be filled by a number.
    :param is_variable: True if the dot *must* be filled by a variable.
    :return: The lsit of dots.
    """
    dots = all_dots(e)
    new_dots = []
    i = 0
    while len(new_dots) != num_requested:
        while DotTerm(i) in dots:
            i = i + 1
        new_dots.append(DotTerm(i, is_numeric, is_variable))
        i = i + 1
    assert len(new_dots) == num_requested
    return new_dots[0] if len(new_dots) == 1 else new_dots


def strip_state(e: Expression) -> None:
    """
    Removes the `state` attribute from a Node. This was originally written so that the tree structure of the AST could
    be directly used as input to pytorch/tensorflow modules while also passing the expressions to "immutable" functions
    that need to make a deep copy of the AST. PyTorch and Tensorflow don't support deep copies when computing gradients.
    :param e: The node to strip state from.
    :return: Nothing; this is a side-effecting function.
    """
    if hasattr(e, "state"):
        e.state = None
    for c in e.children():
        strip_state(c)


def multi_replace(
    replacements: Dict[Expression, Expression], target_input: Expression
) -> Expression:
    """
    Pure function that performs all the replacements in `replacements` in a copy of the `target_input`.
    Warning: does not handle binding properly; these are literal substitutions, not uniform substitutions, and may be
    :param replacements: A mapping what ~> repl where `what` is the expression to replace and `repl` is its replacement.
    :param target_input: The expression on which to perform the replacement.
    :return: A copy of `target_input` with the replacement performed.
    """
    e = target_input.copy()
    for r in replacements:
        # e = replace_without_copy(r, replacements[r], e) would be better?
        e = replace(r, replacements[r], e)  # TODO this is really inefficient.
        if isinstance(e, Formula) != isinstance(target_input, Formula):
            raise AssertionError(
                f"substitution should never rewrite from {target_input.pretty_string()} to {e.pretty_string()}"
            )
    return e


def replace(what: Expression, repl: Expression, target_input: Expression) -> Expression:
    """
    Pure function that replaces all `what`s with `repl`s in a copy of the `target_input`.
    Warning: does not handle binding properly; these are literal substitutions, not uniform substitutions, and may be
    unsound.
    :param what: The expression to replace.
    :param repl: The replacement expression.
    :param target_input: The expression in which the replacement should be made.
    :return: A copy of `target_input` with the replacement performed.
    """
    target = copy.deepcopy(target_input)
    return replace_without_copy(what, repl, target)


def replace_without_copy(
    what: Expression, repl: Expression, target_input: Expression
) -> Expression:
    """
    Side-effecting function that replaces all `what`s with `repl`s in the `target_input`.
    Will print out debugging messages if REPLACEMENT_LOGGING is set to true.
    :param what: The expression to replace.
    :param repl: The replacement expression.
    :param target_input: The expression in which the replacement should be made.
    :return: `target_input`.
    """
    assert isinstance(what, Expression)
    assert isinstance(repl, Expression), f"Expected expression but found {repl}"
    assert isinstance(target_input, Expression)

    if REPLACEMENT_LOGGING:
        replacement_msg = "Replacing %s with %s in %s\n" % (what, repl, target_input)
        logging.info(replacement_msg)

    constructor_stack = []

    def _push_fn(e: Expression):
        if e.is_referentially_eq(what):
            constructor_stack.append(repl)
        else:
            if isinstance(e, (Forall, Exists, Program, DifferentialProgram)):
                raise NotImplementedError(
                    "Substitution only defined for expressions without binding structure."
                )
            constructor_stack.append(e)

    traversal.on_every_node(_push_fn, target_input)
    if REPLACEMENT_LOGGING:
        logging.info(
            f"initial constructor stack for {target_input} is: {constructor_stack}"
        )
    assert target_input in constructor_stack
    assert len(constructor_stack) > 0

    arg_stack = []
    while len(constructor_stack) > 0:
        nxt = constructor_stack.pop()
        if REPLACEMENT_LOGGING:
            logging.info(f"Current constructor stack: {nxt}::{constructor_stack}")
            logging.info(f"Current arg stack: {arg_stack}")
        if isinstance(nxt, (Forall, Exists, PredApp)):
            raise NotImplementedError()
        elif nxt.is_referentially_eq(repl):
            arg_stack.append(nxt)
        elif isinstance(nxt, CompositeExpression):
            assert not isinstance(
                nxt, DotTerm
            ), "not sure why this would happen; just checking."
            if nxt.arity() == 1:
                nxt.__init__(arg_stack.pop())
            elif nxt.arity() == 2:
                assert len(arg_stack) >= 2, (
                    replacement_msg
                    + "About to re-apply arity 2 operator (%s) but only have this arg stack:\n\t%s"
                    % (nxt, arg_stack)
                )
                nxt.__init__(arg_stack.pop(), arg_stack.pop())
            else:
                raise MatchError(
                    replacement_msg
                    + "we now have longer arities that need to be handled."
                )
            arg_stack.append(nxt)
        else:
            arg_stack.append(nxt)

    return arg_stack.pop()


def is_monomial(e: Expression, of: Set[Variable]) -> bool:
    """
    Checks whether an expression is a monomial.
    :param e: The expression to check.
    :param of: The variables of the monomial; others are treated as constants.
    :return: true if e is a monomial of the variables `of`.
    """
    if isinstance(e, (Variable, Number)):
        return True
    elif isinstance(e, Times):
        if not is_monomial(e.left, of) or not is_monomial(e.right, of):
            return False
        # one of the sides needs to not contain any variables in of because x*y^2 is not a monomial in [x,y].
        elif (
            len(of.intersection(variables(e.right))) == 0
            or len(of.intersection(variables(e.left))) == 0
        ):
            return True
        else:
            return False
    elif isinstance(e, Power):
        return (
            is_monomial(e.base, of)
            and (not isinstance(e.base, Power))
            and isinstance(e.exp, Number)
        )
    else:
        return False


def compute_monomials(e: Expression, of: Set[Variable]) -> List[Term]:
    """
    Fins all of the monomials of `of` in `e`.
    :param e: The expression in which to search for monomials.
    :param of: The set of variables of the monomial; others are treated as constants
    :return: A list of all monomials.
    """
    assert len(of) > 0
    rv = []

    def _find_monomial(se: Expression):
        if is_monomial(se, of):
            rv.append(se)

    traversal.on_every_node(_find_monomial, e)
    return rv


def expr_len(e: Expression) -> int:
    """
    Computes the number of nodes in the AST of `e`.
    :param e: The expression whose length is computed.
    :return: The length of the expression.
    """
    return 1 + sum(map(expr_len, e.children()))
