#
# Copyright (C) 2020 IBM. All Rights Reserved.
#
# See LICENSE.txt file in the root directory
# of this source tree for licensing information.
#

import vsrl.verifier.expr as vexpr
from vsrl.parser import parser


class TestParser:
    variables = [
        "a",
        "x",
        "acceleration",
        "relativeAcceleration",
        "x_0",
        "relative_acceleration",
        "x0",
        "x_0_1_1_a",
        "Xrel_0_11_0",
    ]
    numbers = ["1", "1.0", "1.5", "1.25", "0", "-0", "-0.25", "-100.25", "-1"]

    def test_numbers(self):
        for n_str in self.numbers:
            assert vexpr.Number(float(n_str)) == parser.parse(n_str)

    def test_variables(self):
        for v in self.variables:
            assert vexpr.Variable(v) == parser.parse(v)

    def test_neg(self):
        for v in self.variables:
            neg_variable = parser.parse("-" + v)
            assert isinstance(neg_variable, vexpr.Neg)
            assert vexpr.Variable(v) == neg_variable.child
        assert isinstance(parser.parse("-x^(-x)"), vexpr.Neg)

    def test_grouping(self):
        for n_str in self.numbers:
            assert vexpr.Number(float(n_str)) == parser.parse("(" + n_str + ")")

        for v in self.variables:
            assert vexpr.Variable(v) == parser.parse("(" + v + ")")

    def test_power(self):
        assert isinstance(parser.parse("x^2"), vexpr.Power)
        assert isinstance(parser.parse("2^2"), vexpr.Power)
        assert isinstance(parser.parse("x^(2)"), vexpr.Power)
        assert isinstance(parser.parse("(x)^2"), vexpr.Power)
        assert isinstance(parser.parse("(-x)^2"), vexpr.Power)
        assert isinstance(parser.parse("x^-2"), vexpr.Power)
        assert isinstance(parser.parse("x^(-x)"), vexpr.Power)

    def test_divide(self):
        assert isinstance(parser.parse("(x/x)^2"), vexpr.Power)
        assert isinstance(parser.parse("x/x^2"), vexpr.Divide)
        assert isinstance(parser.parse("x / x*y*x"), vexpr.Divide)
        assert isinstance(parser.parse("-x/x/-y/-x"), vexpr.Divide)
        assert isinstance(parser.parse("-(x/x/-y)/-x"), vexpr.Divide)

    def test_mult(self):
        assert isinstance(parser.parse("(x*x)^2"), vexpr.Power)
        assert isinstance(parser.parse("x*x^2"), vexpr.Times)
        assert isinstance(parser.parse("x * x*y*x"), vexpr.Times)
        assert isinstance(parser.parse("-x*x*-y*-x"), vexpr.Times)
        assert isinstance(parser.parse("-(x*x*-y)*-x"), vexpr.Times)

    def test_plus(self):
        assert isinstance(parser.parse("a+b+c+d"), vexpr.Plus)
        assert isinstance(parser.parse("(x + x)^2"), vexpr.Power)
        assert isinstance(parser.parse("x+x^2"), vexpr.Plus)
        assert isinstance(parser.parse("(x*x*y)+x"), vexpr.Plus)
        assert isinstance(parser.parse("-x+-(x*-y*-x)"), vexpr.Plus)
        assert isinstance(parser.parse("(x*x*-y)+-x"), vexpr.Plus)
        assert isinstance(parser.parse("x*y+y*x"), vexpr.Plus)

    def test_minus(self):
        assert isinstance(parser.parse("a-b-c-d"), vexpr.Minus)
        assert isinstance(parser.parse("(x-x)^2"), vexpr.Power)
        assert isinstance(parser.parse("x-x^2"), vexpr.Minus)
        assert isinstance(parser.parse("(x*x*y)-x"), vexpr.Minus)
        assert isinstance(parser.parse("-x--(x*-y*-x)"), vexpr.Minus)
        assert isinstance(parser.parse("(x*x*-y)--x"), vexpr.Minus)
        assert isinstance(parser.parse("x*y-y*x"), vexpr.Minus)

    def test_base_formulas(self):
        asdf = parser.parse("true")
        assert isinstance(asdf, vexpr.TrueF)
        assert isinstance(parser.parse("false"), vexpr.FalseF)
        assert isinstance(parser.parse("a < b"), vexpr.Less)
        assert isinstance(parser.parse("a>b"), vexpr.Greater)
        assert isinstance(parser.parse("a>=   b"), vexpr.GreaterEq)
        assert isinstance(parser.parse("a <= b"), vexpr.LessEq)

    def test_not(self):
        assert isinstance(parser.parse("!false"), vexpr.Not)
        assert isinstance(parser.parse("!   true"), vexpr.Not)

    def test_and(self):
        assert isinstance(parser.parse("(true)&(true)"), vexpr.And)
        assert isinstance(parser.parse("true &  true"), vexpr.And)
        assert isinstance(parser.parse("true &  true & true & true"), vexpr.And)

    def test_whitespaces(self):
        assert isinstance(parser.parse(" a <= b"), vexpr.LessEq)
        assert isinstance(parser.parse(" a"), vexpr.Variable)
        assert isinstance(parser.parse(" a <= b  "), vexpr.LessEq)
        assert isinstance(parser.parse(" a  "), vexpr.Variable)
        assert isinstance(parser.parse("a <= b  "), vexpr.LessEq)
        assert isinstance(parser.parse("a  "), vexpr.Variable)

    def test_imply(self):
        assert isinstance(parser.parse("1=1 -> false -> true"), vexpr.Imply)

    def test_equiv(self):
        assert isinstance(parser.parse("1=1 <-> false"), vexpr.Equiv)

    def test_box(self):
        assert isinstance(parser.parse(" [x:=12 + 2;]  true "), vexpr.Box)
        assert isinstance(parser.parse(" [?false]  true "), vexpr.Box)
        assert isinstance(parser.parse(" [?false;]  true "), vexpr.Box)
        assert isinstance(parser.parse(" [?false++s:=1;]  true "), vexpr.Box)
        assert isinstance(parser.parse(" [?false;++s:=1;]  true "), vexpr.Box)
        assert isinstance(parser.parse(" [?false++s:=1;]  true "), vexpr.Box)
        assert isinstance(parser.parse(" [?false++{s:=1; s:=2;}]  true "), vexpr.Box)

    def test_choice_seq_precedence(self):
        r = parser.parse(" [?false++s:=1; s:=2;]  true ")
        assert isinstance(r.program, vexpr.Choice)
        assert isinstance(r, vexpr.Box)

    def test_loop(self):
        assert isinstance(
            parser.parse(" [{?false++{s:=1; s:=2;}}*]  true ").program, vexpr.Loop
        )
        assert isinstance(
            parser.parse(" 1=1 -> [{?false++{s:=1; s:=2;}}*]  true "), vexpr.Imply
        )
        assert isinstance(
            parser.parse(" [{?false++{s:=1; s:=2;}}*; x:=15]  true "), vexpr.Box
        )
        assert isinstance(
            parser.parse(
                " 1=1 -> [{?false++{s:=1; s:=2;}}*@invariant(1=1&2=2, false)]  true "
            ),
            vexpr.Imply,
        )

    def test_assign(self):
        assert isinstance(parser.parse(" [x:=12 + 2;]  true "), vexpr.Box)

    def test_exists(self):
        assert isinstance(
            parser.parse(r"	  r <= 0 -> \exists f (x=f -> [{x'=r+x^2}]x=f)"),
            vexpr.Imply,
        )

    def test_ode(self):
        assert isinstance(parser.parse("[{x'=1&true}]1=1"), vexpr.Box)
        assert isinstance(parser.parse("[{x'=1,v'=a & true}] 1=1 "), vexpr.Box)
        assert isinstance(parser.parse("[{x'=1, v'=a, z'=12 & true}] 1=1 "), vexpr.Box)
        assert isinstance(
            parser.parse("[{x'=1, v'=a, z'       =    \n  12 & true}] 1=1 "), vexpr.Box
        )
        assert isinstance(
            parser.parse("1=1 -> [{x'=1, v'=a, z'       =    \n  12 & true}] 1=1 "),
            vexpr.Imply,
        )
        assert isinstance(
            parser.parse(
                "1=1 -> [{x'=1, v'=a, z'       =    \n  12 & true}@invariant(true)] 1=1 "
            ),
            vexpr.Imply,
        )

    def test_previously_failed(self):
        assert isinstance(parser.parse("x^2 <= 1/2"), vexpr.Formula)
        assert isinstance(parser.parse("x^2 <= 1/2 & y^2 <= 1/3"), vexpr.Formula)
        assert isinstance(
            parser.parse("1=1 -> [  {x'=1, v'=a, z'       =    \n  12 & true}] 1=1 "),
            vexpr.Imply,
        )
        assert isinstance(
            parser.parse("x^2 <= 1/2 & y^2 <= 1/3 -> [   {x'=-x - (1117*y)/500}]true"),
            vexpr.Imply,
        )
        assert isinstance(
            parser.parse(
                """
            x^2 <= 1/2 & y^2 <= 1/3 -> [   {x'=-x - (1117*y)/500 + (439*y^3)/200 - (333*y^5)/500, y'=x + (617*y)/500 - (439*y^3)/200 + (333*y^5)/500} ] (x - 4*y < 8)
        """
            ),
            vexpr.Formula,
        )
