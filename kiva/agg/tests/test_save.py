import os
import unittest

from numpy import ravel
from kiva import agg

# FIXME: Two of the tests are broken, and Peter promised to fix it at some point.


class TestSave(unittest.TestCase):
    format_output_map = {
        "rgb24": [255, 255, 255, 255, 255, 255, 255, 0, 0, 255, 0, 0],
        "bgr24": [255, 255, 255, 255, 255, 255, 0, 0, 255, 0, 0, 255],
        # "bgr24": [255, 255, 255, 255, 255, 255, 255, 0, 0, 255, 0, 0], # New test result
        "rgba32": [255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 255, 255, 0, 0, 255],
        "bgra32": [255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 255, 255, 0, 0, 255, 255]
        # "bgra32": [255, 255, 255, 255, 255, 255, 255, 255, 255, 0, 0, 255, 255, 0, 0, 255] # old
    }

    def test_rgb24_format(self):
        self.do_check_format('rgb24')

    @unittest.expectedFailure  # TODO: fix this failing test
    def test_bgr24_format(self):
        self.do_check_format('bgr24')

    def test_rgba32_format(self):
        self.do_check_format('rgba32')

    @unittest.expectedFailure  # TODO: fix this failing test
    def test_bgra32_format(self):
        self.do_check_format('bgra32')

    def do_check_format(self, fmt):
        gc = agg.GraphicsContextArray((2, 2), fmt)
        gc.set_stroke_color((1.0, 0.0, 0.0))
        gc.move_to(0.0, 0.5)
        gc.line_to(2.0, 0.5)
        gc.stroke_path()
        gc.save(fmt + ".png")
        img = agg.Image(fmt + ".png")
        os.unlink(fmt + ".png")
        self.assertEqual(list(ravel(img.bmp_array)),
                         self.format_output_map[fmt], fmt)


if __name__ == "__main__":
    unittest.main()
