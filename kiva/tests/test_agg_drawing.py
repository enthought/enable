from kiva.tests.drawing_tester import DrawingImageTester
from kiva.image import GraphicsContext
from traits.testing.unittest_tools import unittest


class TestAggDrawing(DrawingImageTester, unittest.TestCase):

    def create_graphics_context(self, width, height):
        return GraphicsContext((width, height))


if __name__ == "__main__":
    unittest.main()
