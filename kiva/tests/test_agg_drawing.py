import unittest

from kiva.tests.drawing_tester import DrawingImageTester
from kiva.image import GraphicsContext


class TestAggDrawing(DrawingImageTester, unittest.TestCase):

    def setUp(self):
        DrawingImageTester.setUp(self)
        self.gc.set_stroke_color((0.0, 0.0, 0.0))
        self.gc.set_fill_color((0.0, 0.0, 1.0))

    def create_graphics_context(self, width, height):
        return GraphicsContext((width, height))


if __name__ == "__main__":
    unittest.main()
