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

import enable.savage.svg.css.values as values


class FailTest(Exception):
    pass


class ParseTester(object):
    def testValidValues(self):
        # ~ self.parser.debug = True
        try:
            for string, expected in self.valid:
                self.assertEqual(expected, self.parser.parseString(string)[0])
        except ParseException:
            raise FailTest("expected %r to be valid" % string)


class TestInteger(unittest.TestCase, ParseTester):
    parser = values.integer
    valid = [(x, int(x)) for x in ["01", "1"]]


class TestNumber(unittest.TestCase, ParseTester):
    parser = values.number
    valid = [(x, float(x)) for x in ["1.1", "2.3", ".3535"]]
    valid += TestInteger.valid


class TestSignedNumber(unittest.TestCase, ParseTester):
    parser = values.signedNumber
    valid = [(x, float(x)) for x in ["+1.1", "-2.3"]]
    valid += TestNumber.valid


class TestLengthUnit(unittest.TestCase, ParseTester):
    parser = values.lengthUnit
    valid = [(x, x.lower()) for x in ["em", "ex", "px", "PX", "EX", "EM", "%"]]


class TestLength(unittest.TestCase):
    parser = values.length
    valid = [
        ("1.2em", (1.2, "em")),
        ("0", (0, None)),
        ("10045px", (10045, "px")),
        ("300%", (300, "%")),
    ]

    def testValidValues(self):
        for string, expected in self.valid:
            got = self.parser.parseString(string)
            self.assertEqual(expected, tuple(got))

    def testIntegersIfPossible(self):
        results = self.parser.parseString("100px")[0]
        self.assertTrue(isinstance(results, int))

    def testNoSpaceBetweenValueAndUnit(self):
        """ CSS spec section 4.3.2 requires that the
        length identifier immediately follow the value
        """
        self.assertRaises(ParseException, self.parser.parseString, "300 %")
        self.assertRaises(ParseException, self.parser.parseString, "300 px")
