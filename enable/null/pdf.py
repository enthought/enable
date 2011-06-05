#------------------------------------------------------------------------------
# Copyright (c) 2011, Enthought, Inc.
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in enthought/LICENSE.txt and may be redistributed only
# under the conditions described in the aforementioned license.  The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
# Thanks for using Enthought open source!
#------------------------------------------------------------------------------

from kiva.pdf import CompiledPath, GraphicsContext

class NativeScrollBar(object):
    pass

class Window(object):
    pass

def font_metrics_provider():
    from reportlab.pdfgen.canvas import Canvas
    from reportlab.lib.pagesizes import letter
    from kiva.fonttools import Font

    # a file will not be created unless save() is called on the context
    pdf_canvas = Canvas(filename='enable_tmp.pdf', pagesize=letter)
    gc = GraphicsContext(pdf_canvas)
    gc.set_font(Font())
    return gc
