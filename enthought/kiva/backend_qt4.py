#------------------------------------------------------------------------------
# Copyright (c) 2007, Riverbank Computing Limited
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license.
#
# Author: Riverbank Computing Limited
# Description: <Enthought kiva PyQt backend>
#
# In an e-mail to enthought-dev on 2008.09.12 at 2:49 AM CDT, Phil Thompson said:
# The advantage is that all of the PyQt code in ETS can now be re-licensed to
# use the BSD - and I hereby give my permission for that to be done. It's
# been on my list of things to do.
#------------------------------------------------------------------------------
""" This is the Qt backend for kiva. """


# These are the symbols that a backend has to define.
__all__ = ["GraphicsContext", "Canvas", "CompiledPath",
           "font_metrics_provider"]

# Major package imports.
from enthought.qt.api import QtCore, QtGui

# Local imports.
from backend_image import GraphicsContextSystem as GraphicsContext
from agg import CompiledPath
from fonttools import Font


class Canvas(QtGui.QWidget):
    def __init__(self, *args):
        QtGui.QWidget.__init__(self, *args)

        self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent)

        self.clear_color = (1,1,1)
        self.gc = None
        self.new_gc(size=None)

    def do_draw(self, gc):
        """ Method to be implemented by subclasses to actually perform various
        GC drawing commands before the GC is blitted into the screen.
        """
        pass

    def paintEvent(self, event):
        # Clear the gc.
        self.gc.clear(self.clear_color)

        # Draw on the GC.
        self.do_draw(self.gc)

        # Render the pixel data from the graphics context.
        s = self.gc.pixel_map.convert_to_argb32string()
        img = QtGui.QImage(s, self.gc.width(), self.gc.height(),
                QtGui.QImage.Format_ARGB32)

        p = QtGui.QPainter(self)
        p.drawPixmap(0, 0, QtGui.QPixmap.fromImage(img))

    def resizeEvent(self, event):
        width = self.width()
        height = self.height()

        if width != self.gc.width() or height != self.gc.height():
            self.new_gc((width, height))

    def new_gc(self, size):
        """ Creates a new GC of the requested size (or of the widget's current
        size if size is None) and stores it in self.gc.
        """
        if size is None:
            width = self.width()
            height = self.height()
        else:
            width, height = size

        if self.gc is not None:
            del self.gc
            self.gc = None

        self.gc = GraphicsContext((width, height), bottom_up=0)
        

def font_metrics_provider():
    """ Creates an object to be used for querying font metrics.  Typically
    this can just be an empty/dummy graphics context
    """
    gc = GraphicsContext((1,1))
    gc.set_font(Font())
    return gc
