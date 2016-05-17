import contextlib

import six

from kiva.tests.drawing_tester import DrawingTester
from traits.testing.unittest_tools import unittest

try:
    import PyPDF2  # Tests require the PyPDF2 library.
except ImportError:
    PYPDF2_NOT_AVAILABLE = True
else:
    PYPDF2_NOT_AVAILABLE = False

try:
    import reportlab  # noqa
except ImportError:
    REPORTLAB_NOT_AVAILABLE = True
else:
    REPORTLAB_NOT_AVAILABLE = False


@unittest.skipIf(PYPDF2_NOT_AVAILABLE, "PDF tests require PyPDF2")
@unittest.skipIf(REPORTLAB_NOT_AVAILABLE, "Cannot import reportlab")
class TestPDFDrawing(DrawingTester, unittest.TestCase):

    def create_graphics_context(self, width, height):
        from reportlab.pdfgen.canvas import Canvas
        from kiva.pdf import GraphicsContext
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
                line.endswith(six.b('f')),
                line.endswith(six.b('S')),
                line.endswith(six.b('f*')),
                line.endswith(six.b('ET')) and six.b('hello kiva') in line)):
            self.fail('Path was not closed')


if __name__ == "__main__":
    unittest.main()
