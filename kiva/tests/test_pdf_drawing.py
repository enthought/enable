import contextlib
import os
import StringIO
import unittest

import PyPDF2  # Tests require the PyPDF2 library for parsing the pdf stream
from reportlab.pdfgen.canvas import Canvas

from kiva.tests.drawing_tester import DrawingTester
from kiva.pdf import GraphicsContext


class TestPDFDrawing(DrawingTester, unittest.TestCase):

    def setUp(self):
        DrawingTester.setUp(self)

    def create_graphics_context(self, width, height):
        filename = "{0}.pdf".format(self.filename)
        canvas = Canvas(filename, (width, height))
        return GraphicsContext(canvas)

    @contextlib.contextmanager
    def draw_and_check(self):
        yield
        # Save the pdf file.
        filename = "{0}.pdf".format(self.filename)
        self.gc.save()
        reader = PyPDF2.PdfFileReader(filename)
        self.assertEqual(reader.getNumPages(), 1)

        # Find the graphics in the page
        page = reader.getPage(0)
        content = page.getContents()

        # Just a simple check that the path has been closed or the text has
        # been drawn.
        line = content.getData().splitlines()[-2]
        if not any((
                line.endswith('f'),
                line.endswith('S'),
                line.endswith('f*'),
                line.endswith('ET') and 'hello kiva' in line)):
            self.fail('Path was not closed')


if __name__ == "__main__":
    unittest.main()
