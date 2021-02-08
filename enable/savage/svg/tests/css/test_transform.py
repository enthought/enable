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

from enable.savage.svg.css.transform import (
    rotate, scale, skewX, skewY, transform, transformList
)

# list of tuples: parser, string, result
transformTestsGood = [
    (skewY, "skewY(10)", ["skewY", [10]]),
    (skewX, "skewX(10)", ["skewX", [10]]),
    (rotate, "rotate(90)", ["rotate", [90]]),
    (rotate, "rotate(90, 10 10)", ["rotate", [90, 10, 10]]),
    (scale, "scale(.2, .2)", ["scale", [0.2, 0.2]]),
]

# parse, string - exception is always ParseException
transformTestsError = [
    (skewY, "skewY 10"),
    (skewX, "skewX (45"),
    (rotate, "rotate"),
]


class TestTransformParser(unittest.TestCase):
    def testTransformList(self):
        self.assertEqual(
            transformList.parseString(
                "matrix(1,2,3,4,5,6) translate(-10), scale(23, 45.9)"
            ).asList(),
            [
                ["matrix", [1, 2, 3, 4, 5, 6]],
                ["translate", [-10]],
                ["scale", [23, 45.9]],
            ],
        )

    def testTransformGood(self):
        for parser, string, result in transformTestsGood:
            self.assertEqual(transform.parseString(string).asList(), result)

    def testTransformError(self):
        for parser, string in transformTestsError:
            self.assertRaises(ParseException, transform.parseString, string)

    def testPartsGood(self):
        for parser, string, result in transformTestsGood:
            self.assertEqual(parser.parseString(string).asList(), result)

    def testPartsError(self):
        for parser, string in transformTestsError:
            self.assertRaises(ParseException, parser.parseString, string)

    def testMatrixTransform(self):
        src = "matrix(0.966764,0.000000,0.000000,1.062970,-8.322865,-4.427016)"
        expected = [
            [
                "matrix",
                [0.966_764, 0.0, 0.0, 1.062_970, -8.322_865, -4.427_016],
            ]
        ]
        self.assertEqual(transformList.parseString(src).asList(), expected)
