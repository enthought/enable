# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
import sys
import unittest

from kiva.tests.drawing_tester import DrawingImageTester

is_not_macos = not sys.platform == "darwin"


@unittest.skipIf(is_not_macos, "Not macOS")
class TestQuartzDrawing(DrawingImageTester, unittest.TestCase):
    def create_graphics_context(self, width, height, pixel_scale):
        from kiva.quartz import ABCGI

        return ABCGI.CGBitmapContext((width, height), base_pixel_scale=pixel_scale)

    def test_save_dpi(self):
        # Base DPI is 72, but our default pixel scale is 2x.
        self.assertEqual(self.save_and_return_dpi(), 144)
