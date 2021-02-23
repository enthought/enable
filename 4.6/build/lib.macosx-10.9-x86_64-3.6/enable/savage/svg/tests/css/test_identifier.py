# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
import string
import sys
import unittest

from pyparsing import ParseException, Regex, StringEnd

import enable.savage.svg.css.identifier as identifier


class TestEscaped(unittest.TestCase):
    def testEscapedSpecialChar(self):
        for char in string.punctuation:
            self.assertEqual(
                identifier.escaped.parseString("\\" + char)[0], char
            )

    def testEscapeDoesntMatchHexChars(self):
        return
        for char in string.hexdigits:
            self.assertRaises(
                ParseException, identifier.escaped.parseString, "\\" + char
            )


class TestHexUnicode(unittest.TestCase):
    parser = (
        identifier.hex_unicode.copy() + Regex(".+") + StringEnd()
    ).leaveWhitespace()

    def testUnicodeConversion(self):
        self.assertEqual(
            "&", identifier.hex_unicode.parseString(r"\000026")[0]
        )
        self.assertEqual("&", identifier.hex_unicode.parseString(r"\26")[0])

    def testDoesntEatMoreThan6Chars(self):
        self.assertEqual(
            ["&", "B"], list(self.parser.parseString(r"\000026B"))
        )

    def testConsumesFinalSpaceWith6Chars(self):
        self.assertEqual(
            ["&", "B"], list(self.parser.parseString(r"\000026 B"))
        )

    def testConsumesFinalSpaceWithShortChars(self):
        self.assertEqual(["&", "B"], list(self.parser.parseString(r"\26 B")))

    def testDoesntConsumeMoreThanOneSpace(self):
        self.assertEqual(
            ["&", "  B"], list(self.parser.parseString(r"\26   B"))
        )


class TestEscape(unittest.TestCase):
    def testEscapeValues(self):
        self.assertEqual("&", identifier.escape.parseString(r"\26")[0])
        self.assertEqual(
            "\x81", identifier.escape.parseString("\\" + chr(129))[0]
        )
        self.assertEqual("~", identifier.escape.parseString(r"\~")[0])


class TestNonAscii(unittest.TestCase):
    def testNoMatchInAsciiRange(self):
        for c in map(chr, range(128)):
            self.assertRaises(
                ParseException, identifier.nonascii.parseString, c
            )

    def testMatchesOutsideAsciiRange(self):
        for c in map(chr, range(128, sys.maxunicode + 1)):
            self.assertEqual(c, identifier.nonascii.parseString(c)[0])


class TestNmstart(unittest.TestCase):
    def testNmstartValues(self):
        self.assertRaises(ParseException, identifier.nmstart.parseString, "0")


class TestIdentifier(unittest.TestCase):
    def testValidIdentifiers(self):
        for ident in ["import", "color", "border-left"]:
            self.assertEqual(
                ident, identifier.identifier.parseString(ident)[0]
            )
