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

import enable.savage.svg.css.colour as colour


class testColourValueClamping(unittest.TestCase):
    def testByte(self):
        self.assertEqual(100, colour.clampColourByte(100))
        self.assertEqual(0, colour.clampColourByte(-100))
        self.assertEqual(255, colour.clampColourByte(300))


class TestRGBParsing(unittest.TestCase):
    parser = colour.rgb

    def testRGBByte(self):
        self.assertEqual(
            self.parser.parseString("rgb(300,45,100)").asList(),
            ["RGB", [255, 45, 100]],
        )

    def testRGBPerc(self):
        self.assertEqual(
            self.parser.parseString("rgb(100%,0%,0.1%)").asList(),
            ["RGB", [255, 0, 0]],
        )


class TestHexParsing(unittest.TestCase):
    parser = colour.hexLiteral

    def testHexLiteralShort(self):
        self.assertEqual(
            self.parser.parseString("#fab").asList(),
            ["RGB", (0xFF, 0xAA, 0xBB)],
        )

    def testHexLiteralLong(self):
        self.assertEqual(
            self.parser.parseString("#f0a1b2").asList(),
            ["RGB", [0xF0, 0xA1, 0xB2]],
        )

    def testHexLiteralBroken(self):
        badstrings = ["#fab0", "#fab0102d", "#gab"]
        for string in badstrings:
            self.assertRaises(ParseException, self.parser.parseString, string)


class TestNamedColours(unittest.TestCase):
    parser = colour.namedColour

    def testNamedColour(self):
        self.assertEqual(
            self.parser.parseString("fuchsia").asList(),
            ["RGB", (0xFF, 0, 0xFF)],
        )

    def testNamedColourLookupCaseInsensitive(self):
        self.assertEqual(
            self.parser.parseString("fuchsia").asList(),
            self.parser.parseString("FUCHSIA").asList(),
        )


class TestValueParser(TestNamedColours, TestHexParsing, TestRGBParsing):
    parser = colour.colourValue
