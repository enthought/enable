import unittest
import string
import sys
from pyparsing import ParseException, Regex, StringEnd
import enable.savage.svg.css.identifier as identifier

class TestEscaped(unittest.TestCase):
    def testEscapedSpecialChar(self):
        for char in string.punctuation:
            self.assertEqual(
                identifier.escaped.parseString("\\"+char)[0],
                char
            )

    def testEscapeDoesntMatchHexChars(self):
        return
        for char in string.hexdigits:
            self.assertRaises(
                ParseException,
                identifier.escaped.parseString,
                "\\"+char
            )


class TestHexUnicode(unittest.TestCase):
    parser = (identifier.hex_unicode.copy() + Regex(".+") + StringEnd()).leaveWhitespace()
    def testUnicodeConversion(self):
        self.assertEqual(
            u"&",
            identifier.hex_unicode.parseString(r"\000026")[0]
        )
        self.assertEqual(
            u"&",
            identifier.hex_unicode.parseString(r"\26")[0]
        )
    def testDoesntEatMoreThan6Chars(self):
        self.assertEqual(
            [u"&", "B"],
            list(self.parser.parseString(r"\000026B"))
        )
    def testConsumesFinalSpaceWith6Chars(self):
        self.assertEqual(
            [u"&", "B"],
            list(self.parser.parseString(r"\000026 B"))
        )
    def testConsumesFinalSpaceWithShortChars(self):
        self.assertEqual(
            [u"&", "B"],
            list(self.parser.parseString(r"\26 B"))
        )
    def testDoesntConsumeMoreThanOneSpace(self):
        self.assertEqual(
            [u"&", "  B"],
            list(self.parser.parseString(r"\26   B"))
        )


class TestEscape(unittest.TestCase):
    def testEscapeValues(self):
        self.assertEqual(u"&", identifier.escape.parseString(r"\26")[0])
        self.assertEqual(u'\x81', identifier.escape.parseString("\\" + unichr(129))[0])
        self.assertEqual(u"~", identifier.escape.parseString(r'\~')[0])


class TestNonAscii(unittest.TestCase):
    def testNoMatchInAsciiRange(self):
        for c in map(unichr, range(128)):
            self.assertRaises(
                ParseException,
                identifier.nonascii.parseString, c
            )

    def testMatchesOutsideAsciiRange(self):
        for c in map(unichr, xrange(128, sys.maxunicode+1)):
            self.assertEqual(
                c,
                identifier.nonascii.parseString(c)[0]
            )

class TestNmstart(unittest.TestCase):
    def testNmstartValues(self):
        self.assertRaises(
            ParseException,
            identifier.nmstart.parseString,
            "0"
        )


class TestIdentifier(unittest.TestCase):
    def testValidIdentifiers(self):
        for ident in ["import","color", "border-left"]:
            self.assertEqual(
                ident,
                identifier.identifier.parseString(ident)[0]
            )
