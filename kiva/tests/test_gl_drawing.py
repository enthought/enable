from kiva.tests.drawing_tester import DrawingImageTester
from kiva.gl import GraphicsContext
from traits.testing.unittest_tools import unittest


class TestGLDrawing(DrawingImageTester, unittest.TestCase):

    def create_graphics_context(self, width, height):
        return GraphicsContext((width, height))

    def test_star_clip(self):
        # FIXME: overriding test since it segfaults
        self.fail('This normally segfaults')

if __name__ == "__main__":
    unittest.main()
