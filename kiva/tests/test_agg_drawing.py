import unittest

from kiva.tests.drawing_tester import DrawingTester
from kiva.image import GraphicsContext


class TestAggDrawing(DrawingTester, unittest.TestCase):

    def setUp(self):
        DrawingTester.setUp(self)
        self.gc.set_stroke_color((0.0, 0.0, 0.0))
        self.gc.set_fill_color((0.0, 0.0, 1.0))

    def create_graphics_context(self, width, height):
        return GraphicsContext((width, height))

    def save_to_file(self):
        self.gc.save(self.filename)


if __name__ == "__main__":
    unittest.main()
