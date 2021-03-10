# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen.canvas import Canvas

import kiva.pdf as pdf_backend

# Pass it along
CompiledPath = pdf_backend.CompiledPath


class GraphicsContext(pdf_backend.GraphicsContext):
    """ This is a wrapper of the PDF GraphicsContext which works with the
    benchmark program.
    """
    def __init__(self, size, *args, **kw):
        canvas = Canvas('', pagesize=letter)
        super().__init__(canvas, *args, **kw)

    def save(self, filename, *args, **kw):
        # Reportlab is a bit silly
        self.gc._filename = filename
        super().save(filename, *args, **kw)
