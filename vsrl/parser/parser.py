#
# Copyright (C) 2020 IBM. All Rights Reserved.
#
# See LICENSE.txt file in the root directory
# of this source tree for licensing information.
#

import functools
from typing import Optional, Union

import parsimonious
from parsimonious.grammar import Grammar

import vsrl.verifier.expr as vexpr

_grammar = Grammar(
    r"""
    expression = formula / term / program

    term = ~r"\s*"? (minus / plus / divide / mult / power / number / variable / neg / term_group) ~r"\s*"?
    term_group = "(" term ")"
    number = ~r"-?[0-9]+(\.[0-9]+)?"
    variable = ~"(?!true)([a-zA-Z][a-zA-Z0-9_]*)"
    neg = "-" (power / term_group / number / variable / neg)
    power = (number / variable / term_group) ((~r"\s*\^\s*") (number / variable / term_group))+
    mult = (power / number / variable / neg / term_group) ((~r"\s*\*\s*") (power / number / variable / neg / term_group))+
    divide = (mult / power / number / variable / neg / term_group) ((~r"\s*/\s*") (mult / power / number / variable / neg / term_group))+
    plus = (divide / mult / power / number / variable / neg / term_group) ((~r"\s*\+\s*") (divide / mult / power / number / variable / neg / term_group))+
    minus = (plus / divide / mult / power / number / variable / neg / term_group) ((~r"\s*-\s*") (plus / divide / mult / power / number / variable / neg / term_group))+

    formula = ~r"\s*"? (forall / exists / equiv / imply / box / or / and / not / greatereq / lesseq / less / greater / equal / true / false / formula_group) ~r"\s*"?
    formula_group = "(" formula ")"
    true = ~"true"
    false = ~"false"
    less = term (~r"\s*<\s*") term
    lesseq = term (~r"\s*<=\s*") term
    greater = term (~r"\s*>\s*") term
    greatereq = term (~r"\s*>=\s*") term
    equal = term (~r"\s*=\s*") term
    not = (~r"\s*\!\s*") (forall / exists / greatereq / lesseq / less / greater / equal / true / false / formula_group)
    and = (not / greatereq / lesseq / less / greater / equal / true / false / formula_group) ((~r"\s*&\s*") (forall / exists / not / greatereq / lesseq / less / greater / equal / true / false / formula_group))+
    or = (and / not / greatereq / lesseq / less / greater / equal / true / false / formula_group) ((~r"\s*\|\s*") (forall / exists / and / not / greatereq / lesseq / less / greater / equal / true / false / formula_group))+
    imply = (box / or / and / not / greatereq / lesseq / less / greater / equal / true / false / formula_group) ((~r"\s*" "->" ~r"\s*") (forall / exists / box / or / and / not / greatereq / lesseq / less / greater / equal / true / false / formula_group))+
    equiv = (imply / box / or / and / not / greatereq / lesseq / less / greater / equal / true / false / formula_group) ((~r"\s*\<-\>\s*") (forall / exists / imply / box / or / and / not / greatereq / lesseq / less / greater / equal / true / false / formula_group))+
    box = (~r"\s*\[\s*") program (~r"\s*\]\s*") formula (~r"\s*")

    variable_list = variable / (("{" ~r"\s*")? variable+ (~r"\s*" "}")?)
    forall = ("\\forall" ~r"\s*") variable_list (~r"\s*" "."? ~r"\s*") formula
    exists = ("\\exists" ~r"\s*") variable_list (~r"\s*" "."? ~r"\s*") formula

    program = ~r"\s*"? (seqcomp / choice / star / test / ode_system / ode_system_no_evdom / assignment / program_group) ~r"\s*;\s*"?
    program_group = (~r"\s*\{\s*") program (~r"\s*\}\s*")
    assignment = variable (~r"\s*:=\s*") term
    test = (~r"\s*" "?" ~r"\s*") formula
    seqcomp = (star / test / ode_system / assignment / program_group) ((~r"\s*" ";" ~r"\s*") (star / test / ode_system / assignment / program_group))+
    choice = (star / seqcomp / test / ode_system / assignment / program_group) ((~r"\s*" ~r"\s*;\s*"? "++" ~r"\s*") (star / seqcomp / test / ode_system / assignment / program_group) ~r"\s*;\s*"?)+

    inv_annotation = ~r"\s*" "@invariant" (~r"\s*" "(" ~r"\s*") formula (~r"\s*" "," ~r"\s*" formula)* (~r"\s*" ")" ~r"\s*")
    star = (~r"\s*\{\s*") (seqcomp / choice / test / ode_system / assignment / program_group) (~r"\s*\}\s*") (~r"\s*\*\s*") inv_annotation?

    atomic_ode = ~r"\s*" variable ~r"'\s*=\s*" term ~r"\s*"
    diff_product = atomic_ode (~r"\s*,\s*" atomic_ode)*
    ode_system = (~r"\s*\{\s*") (diff_product / atomic_ode) (~r"\s*&\s*") formula ~r"\s*\}\s*" inv_annotation?
    ode_system_no_evdom = (~r"\s*\{\s*") (diff_product / atomic_ode) ~r"\s*\}\s*" inv_annotation?
    """
)


class ParserException(Exception):
    def __init__(self, msg: str, info: Optional[dict] = None):
        self.msg = msg
        self.info = info or {}

    def __str__(self):
        return self.msg


def _collect_right_assoc(
    node: parsimonious.nodes.Node,
) -> Union[vexpr.Expression, vexpr.Formula, vexpr.Program]:
    rv = [_convert(node.children[0])]
    for child in node.children[1]:
        assert len(child.children) == 2 or len(child.children) == 3
        the_expr = child.children[1]
        rv.append(_convert(the_expr))
    return rv


def _convert(ast: parsimonious.nodes.Node) -> vexpr.Expression:
    if ast.expr_name == "term":
        assert len(ast.children) == 3
        return _convert(ast.children[1])
    elif ast.expr_name == "formula":
        assert len(ast.children) == 3
        return _convert(ast.children[1])
    elif ast.expr_name == "program":
        assert len(ast.children) == 3
        return _convert(ast.children[1])
    if ast.expr_name == "number":
        return vexpr.Number(float(ast.text))
    elif ast.expr_name == "variable":
        return vexpr.Variable(ast.text)
    elif ast.expr_name == "neg":
        assert len(ast.children) == 2
        return vexpr.Neg(_convert(ast.children[1]))
    elif (
        ast.expr_name == "term_group"
        or ast.expr_name == "formula_group"
        or ast.expr_name == "program_group"
    ):
        assert len(ast.children) == 3
        return _convert(ast.children[1])
    elif ast.expr_name == "power":
        return functools.reduce(vexpr.Power, _collect_right_assoc(ast))
    elif ast.expr_name == "mult":
        return functools.reduce(vexpr.Times, _collect_right_assoc(ast))
    elif ast.expr_name == "divide":
        return functools.reduce(vexpr.Divide, _collect_right_assoc(ast))
    elif ast.expr_name == "plus":
        return functools.reduce(vexpr.Plus, _collect_right_assoc(ast))
    elif ast.expr_name == "minus":
        return functools.reduce(vexpr.Minus, _collect_right_assoc(ast))
    elif ast.expr_name == "true":
        return vexpr.TrueF()
    elif ast.expr_name == "false":
        return vexpr.FalseF()
    elif ast.expr_name == "not":
        return vexpr.Not(_convert(ast.children[1]))
    elif ast.expr_name == "less":
        left = _convert(ast.children[0])
        right = _convert(ast.children[2])
        return vexpr.Less(left, right)
    elif ast.expr_name == "lesseq":
        left = _convert(ast.children[0])
        right = _convert(ast.children[2])
        return vexpr.LessEq(left, right)
    elif ast.expr_name == "greater":
        left = _convert(ast.children[0])
        right = _convert(ast.children[2])
        return vexpr.Greater(left, right)
    elif ast.expr_name == "greatereq":
        left = _convert(ast.children[0])
        right = _convert(ast.children[2])
        return vexpr.GreaterEq(left, right)
    elif ast.expr_name == "equal":
        left = _convert(ast.children[0])
        right = _convert(ast.children[2])
        return vexpr.Eq(left, right)
    elif ast.expr_name == "and":
        return functools.reduce(vexpr.And, _collect_right_assoc(ast))
    elif ast.expr_name == "or":
        return functools.reduce(vexpr.Or, _collect_right_assoc(ast))
    elif ast.expr_name == "imply":
        return functools.reduce(vexpr.Imply, _collect_right_assoc(ast))
    elif ast.expr_name == "equiv":
        return functools.reduce(vexpr.Equiv, _collect_right_assoc(ast))
    elif ast.expr_name == "box":
        return vexpr.Box(_convert(ast.children[1]), _convert(ast.children[3]))
    elif ast.expr_name == "variable_list":
        # variable_list = variable / (("{" ~"\s*")? variable+ (~"\s*" "}")?)
        if len(ast.children) == 1:
            return [_convert(ast.children[0])]
        else:
            vs = []
            for c in ast.children[0].children[1]:
                vs.append(_convert(c))
            return vs
    elif ast.expr_name == "forall":
        vs = _convert(ast.children[1])
        f = _convert(ast.children[3])
        return vexpr.Forall(set(vs), f)
    elif ast.expr_name == "exists":
        vs = _convert(ast.children[1])
        f = _convert(ast.children[3])
        return vexpr.Forall(set(vs), f)
    elif ast.expr_name == "assignment":
        return vexpr.Assign(_convert(ast.children[0]), _convert(ast.children[2]))
    elif ast.expr_name == "test":
        return vexpr.Test(_convert(ast.children[1]))
    elif ast.expr_name == "star":
        return vexpr.Loop(_convert(ast.children[1]))
    elif ast.expr_name == "choice":
        return functools.reduce(vexpr.Choice, _collect_right_assoc(ast))
    elif ast.expr_name == "seqcomp":
        return functools.reduce(vexpr.SequentialCompose, _collect_right_assoc(ast))
    elif ast.expr_name == "assignment":
        return vexpr.Box(_convert(ast.children[0]), _convert(ast.children[2]))
    elif ast.expr_name == "atomic_ode":
        return vexpr.AtomicODE(_convert(ast.children[1]), _convert(ast.children[3]))
    elif ast.expr_name == "diff_product":
        return functools.reduce(vexpr.DiffPair, _collect_right_assoc(ast))
    elif ast.expr_name == "ode_system":
        f = _convert(ast.children[3])
        ode = ast.children[1]
        assert len(ode.children) == 1
        ode = ode.children[0]
        assert ode.expr_name == "diff_product" or ode.expr_name == "atomic_ode"
        return vexpr.ODESystem(_convert(ode), f)
    elif ast.expr_name == "ode_system_no_evdom":
        ode = ast.children[1]
        assert len(ode.children) == 1
        ode = ode.children[0]
        assert ode.expr_name == "diff_product" or ode.expr_name == "atomic_ode"
        return vexpr.ODESystem(_convert(ode), vexpr.TrueF())
    else:
        if len(ast.children) != 1:
            raise ParserException(
                f"Successfully parsed into a parsimonious library node, but "
                f"_convert does not know how to handle expr_name: {ast.expr_name}"
            )
        else:
            return _convert(ast.children[0])


def parse(s: str) -> vexpr.Expression:
    ast = _grammar.parse(s)
    return _convert(ast)
