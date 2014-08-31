import unittest

from kiva.tests.drawing_tester import DrawingTester
from kiva.pdf import GraphicsContext
from reportlab.pdfgen.canvas import Canvas


class TestPDFDrawing(DrawingTester, unittest.TestCase):

    def setUp(self):
        DrawingTester.setUp(self)
        self.gc.set_stroke_color((0.0, 0.0, 0.0))
        self.gc.set_fill_color((0.0, 0.0, 1.0))

    def create_graphics_context(self, width, height):
        canvas = Canvas(self.filename, (width, height))
        return GraphicsContext(canvas)

    def save_to_file(self):
        self.gc.save()


if __name__ == "__main__":
    unittest.main()
