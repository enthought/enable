from kiva.tests.drawing_tester import DrawingImageTester
from traits.testing.unittest_tools import unittest

try:
    import cairo  # noqa
except ImportError:
    CAIRO_NOT_AVAILABLE = True
else:
    CAIRO_NOT_AVAILABLE = False


@unittest.skipIf(CAIRO_NOT_AVAILABLE, "Cannot import cairo")
class TestCairoDrawing(DrawingImageTester, unittest.TestCase):

    def create_graphics_context(self, width, height):
        from kiva.cairo import GraphicsContext
        return GraphicsContext((width, height))


if __name__ == "__main__":
    unittest.main()
