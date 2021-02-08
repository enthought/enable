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

import enable.savage.svg.attributes as a

from .css.test_color import TestValueParser


class TestURLParser(unittest.TestCase):
    parser = a.url

    def testURL(self):
        self.assertEqual(
            self.parser.parseString("url(#someGradient)").asList(),
            ["URL", [("", "", "", "", "someGradient"), ()]],
        )

    def testURLWithFallback(self):
        self.assertEqual(
            self.parser.parseString("url(someGradient) red").asList(),
            ["URL", [("", "", "someGradient", "", ""), ["RGB", (255, 0, 0)]]],
        )

    def testEmptyURLWithFallback(self):
        self.assertEqual(
            self.parser.parseString("url() red").asList(),
            ["URL", [("", "", "", "", ""), ["RGB", (255, 0, 0)]]],
        )

    def testEmptyURL(self):
        self.assertEqual(
            self.parser.parseString("url()").asList(),
            ["URL", [("", "", "", "", ""), ()]],
        )

    def testxPointerURL(self):
        self.assertEqual(
            self.parser.parseString("url(#xpointer(idsomeGradient))").asList(),
            ["URL", [("", "", "", "", "xpointer(idsomeGradient)"), ()]],
        )


class TestPaintValueURL(TestURLParser):
    parser = a.paintValue


class TestPaintValue(TestValueParser):
    parser = a.paintValue

    def testNone(self):
        self.assertEqual(
            self.parser.parseString("none").asList(), ["NONE", ()]
        )

    def testCurrentColor(self):
        self.assertEqual(
            self.parser.parseString("currentColor").asList(),
            ["CURRENTCOLOR", ()],
        )
