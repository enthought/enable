# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
import contextlib
import unittest

from kiva.tests.drawing_tester import DrawingTester

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
        if not any((line.endswith(b'f'),
                    line.endswith(b'Q'),
                    line.endswith(b'S'),
                    line.endswith(b'f*'),
                    line.endswith(b'ET') and b'hello kiva' in line)):
            self.fail('Path was not closed')
