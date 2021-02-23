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

from kiva.agg import AggFontType, GraphicsContextArray
from kiva.api import Font


class UnicodeTest(unittest.TestCase):
    def test_show_text_at_point(self):
        gc = GraphicsContextArray((100, 100))
        gc.set_font(Font())
        gc.show_text_at_point(str("asdf"), 5, 5)

    def test_agg_font_type(self):
        f1 = AggFontType("Arial")
        f2 = AggFontType(b"Arial")
        self.assertEqual(f1, f2)
