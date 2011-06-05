import os
import unittest

from numpy import allclose, ravel

import nose

from kiva import agg


# FIXME:
#   These tests are broken, and Peter promised to fix it at some point.

class Test_Save(unittest.TestCase):
    format_output_map = {
        "rgb24": [255,255,255,255,255,255,255,0,0,255,0,0],
        "bgr24": [255,255,255,255,255,255,0,0,255,0,0,255],
        "rgba32": [255,255,255,255,255,255,255,255,255,0,0,255,255,0,0,255],
        "bgra32": [255,255,255,255,255,255,255,255,0,0,255,255,0,0,255,255]
        }

    def test_rgb24_format(self):
        self.do_check_format('rgb24')

    def test_bgr24_format(self):
        self.do_check_format('bgr24')

    def test_rgba32_format(self):
        self.do_check_format('rgba32')

    def test_bgra32_format(self):
        self.do_check_format('bgra32')

    def do_check_format(self,fmt):
        # FIXME:
        raise nose.SkipTest

        gc = agg.GraphicsContextArray((2,2), fmt)
        gc.set_stroke_color((1.0,0.0,0.0))
        gc.move_to(0.0, 0.5)
        gc.line_to(2.0, 0.5)
        gc.stroke_path()
        gc.save(fmt + ".png")
        img = agg.Image(fmt + ".png")
        os.unlink(fmt + ".png")
        self.assertEqual(list(ravel(img.bmp_array)),
                         self.format_output_map[fmt])

if __name__ == "__main__":
    unittest.main()
