from enthought.util.numerix import *
from enthought.util.testingx import *
import unittest
import os

from enthought.kiva import agg

class TestSave(unittest.TestCase):
    format_output_map = {
        "rgb24": [255,255,255,255,255,255,255,0,0,255,0,0],
        "bgr24": [255,255,255,255,255,255,0,0,255,0,0,255],
        "rgba32": [255,255,255,255,255,255,255,255,255,0,0,255,255,0,0,255],
        "bgra32": [255,255,255,255,255,255,255,255,0,0,255,255,0,0,255,255]
        }
    def check_rgb24_format(self):
        self.do_check_format('rgb24')
    def check_bgr24_format(self):
        self.do_check_format('bgr24')
    def check_rgba32_format(self):
        self.do_check_format('rgba32')
    def check_bgra32_format(self):
        self.do_check_format('bgra32')
    def do_check_format(self,fmt):
        gc = agg.GraphicsContextArray((2,2), fmt)
        gc.set_stroke_color((1.0,0.0,0.0))
        gc.move_to(0.0, 0.5)
        gc.line_to(2.0, 0.5)
        gc.stroke_path()
        gc.save(fmt + ".png")
        img = agg.Image(fmt + ".png")
        os.unlink(fmt + ".png")
        assert_equal(ravel(img.bmp_array),self.format_output_map[fmt])

#----------------------------------------------------------------------------
# test setup code.
#----------------------------------------------------------------------------

def check_suite(level=1):
    suites = []
    if level > 0:
        suites.append( unittest.makeSuite(TestSave,'check_') )
    total_suite = unittest.TestSuite(suites)
    return total_suite

def test(level=10):
    all_tests = check_suite(level)
    runner = unittest.TextTestRunner()
    runner.run(all_tests)
    return runner

if __name__ == "__main__":
    test()
