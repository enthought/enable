import unittest

import numpy as np

from kiva.tests.drawing_tester import DrawingImageTester
from kiva.image import GraphicsContext


class TestAggDrawing(DrawingImageTester, unittest.TestCase):

    def create_graphics_context(self, width, height):
        return GraphicsContext((width, height))

    def test_unicode_gradient_args(self):
        color_nodes = [(0.0, 1.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)]
        with self.draw_and_check():
            w, h = self.gc.width(), self.gc.height()
            grad_stops = np.array([(x, r, g, b, 1.0)
                                   for x, r, g, b in color_nodes])

            self.gc.rect(0, 0, w, h)
            self.gc.linear_gradient(0, 0, w, 0, grad_stops,
                                    u'pad', b'userSpaceOnUse')
            self.gc.fill_path()


if __name__ == "__main__":
    unittest.main()
