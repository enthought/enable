# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
import unittest

from pyparsing import ParseException

from enable.savage.svg.pathdata import (
    Sequence, closePath, coordinatePair, curve, ellipticalArc, horizontalLine,
    lineTo, moveTo, number, quadraticBezierCurveto,
    smoothQuadraticBezierCurveto, svg, verticalLine
)


class TestNumber(unittest.TestCase):
    parser = number
    valid = ["1.e10", "1e2", "1e+4", "1e-10", "1.", "1.0", "0.1", ".2"]
    invalid = ["e10", ".", "f", ""]

    def testValid(self):
        for num in self.valid:
            self.assertEqual(float(num), self.parser.parseString(num)[0])

    def testInvalid(self):
        for num in self.invalid:
            self.assertRaises(
                ParseException, lambda: self.parser.parseString(num)
            )


class TestNumberSequence(unittest.TestCase):
    def testFloatsWithNoSpacing(self):
        self.assertEqual(
            [0.4, 0.4], list(Sequence(number).parseString("0.4.4"))
        )


class TestCoords(unittest.TestCase):
    def testCoordPair(self):
        self.assertEqual(
            coordinatePair.parseString("100 100")[0], (100.0, 100.0)
        )
        self.assertEqual(coordinatePair.parseString("100,2E7")[0], (100, 2e7))

    def testCoordPairWithMinus(self):
        self.assertEqual(
            coordinatePair.parseString("100-100")[0], (100.0, -100.0)
        )

    def testCoordPairWithPlus(self):
        self.assertEqual(
            coordinatePair.parseString("100+100")[0], (100.0, 100.0)
        )

    def testCoordPairWithPlusAndExponent(self):
        self.assertEqual(
            coordinatePair.parseString("100+1e+2")[0], (100.0, 100.0)
        )

    def testNotAPair(self):
        self.assertRaises(ParseException, coordinatePair.parseString, "100")
        self

    def testNoSpacing(self):
        self.assertEqual(coordinatePair.parseString("-1.1.1")[0], (-1.1, 0.1))


class TestMoveTo(unittest.TestCase):
    def testSimple(self):
        self.assertEqual(
            moveTo.parseString("M 100 100").asList()[0],
            ["M", [(100.0, 100.0)]],
        )

    def testLonger(self):
        self.assertEqual(
            moveTo.parseString("m 100 100 94 1e7").asList()[0],
            ["m", [(100.0, 100.0), (94, 1e7)]],
        )

    def testLine(self):
        self.assertEqual(
            lineTo.parseString("l 300 100").asList()[0],
            ["l", [(300.0, 100.0)]],
        )

    def testHorizonal(self):
        self.assertEqual(
            horizontalLine.parseString("H 100 5 20").asList()[0],
            ["H", [100.0, 5.0, 20.0]],
        )

    def testVertical(self):
        self.assertEqual(
            verticalLine.parseString("V 100 5 20").asList()[0],
            ["V", [100.0, 5.0, 20.0]],
        )


class TestEllipticalArc(unittest.TestCase):
    def testParse(self):
        self.assertEqual(
            ellipticalArc.parseString("a25,25 -30 0,1 50,-25").asList()[0],
            ["a", [[(25.0, 25.0), -30.0, (False, True), (50.0, -25.0)]]],
        )

    def testExtraArgs(self):
        self.assertEqual(
            ellipticalArc.parseString(
                "a25,25 -30 0,1 50,-25, 10, 10"
            ).asList()[0],
            ["a", [[(25.0, 25.0), -30.0, (False, True), (50.0, -25.0)]]],
        )


class TestSmoothQuadraticBezierCurveto(unittest.TestCase):
    def testParse(self):
        self.assertEqual(
            smoothQuadraticBezierCurveto.parseString("t1000,300").asList()[0],
            ["t", [(1000.0, 300.0)]],
        )


class TestQuadraticBezierCurveto(unittest.TestCase):
    def testParse(self):
        self.assertEqual(
            quadraticBezierCurveto.parseString("Q1000,300 200 5").asList()[0],
            ["Q", [[(1000.0, 300.0), (200.0, 5.0)]]],
        )


class TestCurve(unittest.TestCase):
    def testParse(self):
        self.assertEqual(
            curve.parseString(
                "C 100 200 300 400 500 600 100 200 300 400 500 600"
            ).asList()[0],
            [
                "C",
                [
                    [(100.0, 200.0), (300.0, 400.0), (500.0, 600.0)],
                    [(100.0, 200.0), (300.0, 400.0), (500.0, 600.0)],
                ],
            ],
        )


class TestClosePath(unittest.TestCase):
    def testParse(self):
        self.assertEqual(
            closePath.parseString("Z").asList()[0], ("Z", (None,))
        )


class TestSVG(unittest.TestCase):
    def testParse(self):
        path = ("M 100 100 L 300 100 L 200 300 z a 100,100 -4 0,1 25 25 "
                "z T300 1000 t40, 50 h4 42 2 2,1v1,1,1 Z Q 34,10 1 1")
        r = svg.parseString(path).asList()
        expected = [
            ["M", [(100, 100)]],
            ["L", [(300, 100)]],
            ["L", [(200, 300)]],
            ("Z", (None,)),
            ["a", [[(100, 100), -4, (False, True), (25, 25)]]],
            ("Z", (None,)),
            ["T", [(300, 1000)]],
            ["t", [(40, 50)]],
            ["h", [4, 42, 2, 2, 1]],
            ["v", [1, 1, 1]],
            ("Z", (None,)),
            ["Q", [[(34, 10), (1, 1)]]],
        ]
        self.assertEqual(len(r), len(expected))
        for a, b in zip(expected, r):
            self.assertEqual(a, b)
