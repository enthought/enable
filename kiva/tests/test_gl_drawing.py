import unittest

from kiva.tests.drawing_tester import DrawingImageTester
from kiva.gl import GraphicsContext


class TestGLDrawing(DrawingImageTester, unittest.TestCase):

    def create_graphics_context(self, width, height):
        return GraphicsContext((width, height))


if __name__ == "__main__":
    unittest.main()
