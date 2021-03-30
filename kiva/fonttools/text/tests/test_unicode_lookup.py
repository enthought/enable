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

from kiva.fonttools.text._unicode_lookup import UnicodeAnalyzer


class TestUnicodeAnalyzer(unittest.TestCase):
    def test_sample_strings(self):
        an = UnicodeAnalyzer()

        st = "Hello World"
        res = an.languages(st)
        self.assertListEqual(res, [(0, len(st), "Common")])

        st = "你好世界"
        res = an.languages(st)
        self.assertListEqual(res, [(0, len(st), "Han")])

        st = "Kiva Graphics一番😎"
        expected = [(0, 13, "Common"), (13, 15, "Han"), (15, 16, 'Common')]
        res = an.languages(st)
        self.assertListEqual(res, expected)
