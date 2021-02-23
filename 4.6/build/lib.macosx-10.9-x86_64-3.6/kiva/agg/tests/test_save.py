# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
import os
import unittest

from numpy import ravel

from kiva import agg


# FIXME:
#   These tests are broken, and Peter promised to fix it at some point.
#   see enthought/enable#480
@unittest.skip("tests are broken, see enthought/enable#480")
class Test_Save(unittest.TestCase):
    format_output_map = {
        "rgb24": [255, 255, 255, 255, 255, 255, 255, 0, 0, 255, 0, 0],
        "bgr24": [255, 255, 255, 255, 255, 255, 0, 0, 255, 0, 0, 255],
        "rgba32": [255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 255,
                   255, 0, 0, 255],
        "bgra32": [255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 255, 255,
                   0, 0, 255, 255],
    }

    def test_rgb24_format(self):
        self.do_check_format("rgb24")

    def test_bgr24_format(self):
        self.do_check_format("bgr24")

    def test_rgba32_format(self):
        self.do_check_format("rgba32")

    def test_bgra32_format(self):
        self.do_check_format("bgra32")

    def do_check_format(self, fmt):
        # FIXME:

        gc = agg.GraphicsContextArray((2, 2), fmt)
        gc.set_stroke_color((1.0, 0.0, 0.0))
        gc.move_to(0.0, 0.5)
        gc.line_to(2.0, 0.5)
        gc.stroke_path()
        gc.save(fmt + ".png")
        img = agg.Image(fmt + ".png")
        os.unlink(fmt + ".png")
        self.assertEqual(
            list(ravel(img.bmp_array)), self.format_output_map[fmt]
        )
