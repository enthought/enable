import contextlib
import StringIO
import unittest

import PyPDF2  # Tests require the PyPDF2 library for parsing the pdf stream
from reportlab.pdfgen.canvas import Canvas

from kiva.tests.drawing_tester import DrawingTester
from kiva.pdf import GraphicsContext


class TestPDFDrawing(DrawingTester, unittest.TestCase):

    def setUp(self):
        DrawingTester.setUp(self)
        self.gc.set_stroke_color((0.0, 0.0, 0.0))
        self.gc.set_fill_color((0.0, 0.0, 1.0))

    def create_graphics_context(self, width, height):
        canvas = Canvas(self.filename, (width, height))
        return GraphicsContext(canvas)

    @contextlib.contextmanager
    def draw_and_check(self):
        yield
        # Read the pdfstream.
        pdfdata = self.gc.gc.getpdfdata()
        stream = StringIO.StringIO(pdfdata)
        reader = PyPDF2.PdfFileReader(stream)
        self.assertEqual(reader.getNumPages(), 1)

        # Find the graphics in the page
        page = reader.getPage(0)
        content = page.getContents()

        # Just a simple check that the path has been closed or the text has
        # been drawn.
        line = content.getData().splitlines()[-2].strip()
        if not any((
            line.endswith('f'),
            line.endswith('S'),
            line.endswith('f*'),
            line.endswith('ET') and 'hello kiva' in line)):
            self.fail('Path was not closed')


if __name__ == "__main__":
    unittest.main()
