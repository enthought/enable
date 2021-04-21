# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
""" Defines the concrete top-level Enable 'Window' class for the wxPython GUI
toolkit, based on the Quartz kiva backend for OS X.
"""

# Major library imports.
import numpy as np
import wx

# Enthought library imports.
from kiva.api import Font
from kiva.quartz import get_macport, ABCGI

# Local imports.
from .base_window import BaseWindow
from .scrollbar import NativeScrollBar

CompiledPath = ABCGI.CGMutablePath


class GraphicsContext(ABCGI.CGLayerContext):
    def __init__(self, size_or_array, *args, **kwds):
        gc = kwds.pop("window_gc", None)
        if not gc:
            # Create a tiny base context to spawn the CGLayerContext from.
            # We are better off making our Layer from the window gc since
            # the data formats will match and so it will be faster to draw the
            # layer.
            gc = ABCGI.CGBitmapContext((1, 1))
        if isinstance(size_or_array, np.ndarray):
            # Initialize the layer with an image.
            image = ABCGI.CGImage(size_or_array)
            width = image.width
            height = image.height
        else:
            # No initialization.
            image = None
            width, height = size_or_array

        super().__init__((width, height), gc, *args, **kwds)
        if image is not None:
            self.draw_image(image)

    @classmethod
    def create_from_gc(klass, gc, size_or_array, *args, **kwds):
        return klass(size_or_array, window_gc=gc, *args, **kwds)


class _WindowGraphicsContext(ABCGI.CGContextInABox):
    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)
        self._begun = False

    def begin(self):
        if self._begun:
            return
        self.save_state()
        self.translate_ctm(0, self.height())
        self.scale_ctm(1.0, -1.0)
        self._begun = True

    def end(self):
        if self._begun:
            self.restore_state()
            self._begun = False

    @staticmethod
    def create_from_gc(gc, size_or_array, *args, **kwds):
        return GraphicsContext(size_or_array, window_gc=gc, *args, **kwds)


class Window(BaseWindow):
    """ An Enable Window for wxPython GUIs on OS X.
    """

    # 'BaseWindow' interface ################################################

    def _create_gc(self, size, pix_format="bgra32"):
        self.dc = wx.ClientDC(self.control)
        gc = _WindowGraphicsContext(self.dc.GetSize(), get_macport(self.dc))
        gc.begin()
        return gc

    def _window_paint(self, event):
        self.dc = None
        self._gc = None  # force a new gc to be created for the next paint()

    # 'AbstractWindow' interface ############################################

    def _paint(self, event=None):
        size = self._get_control_size()
        if (self._size != tuple(size)) or (self._gc is None):
            self._gc = self._create_gc(size)
            self._size = tuple(size)
        gc = self._gc
        gc.begin()
        gc.clear(self.bgcolor_)
        if hasattr(self.component, "do_layout"):
            self.component.do_layout()
        self.component.draw(gc, view_bounds=(0, 0, size[0], size[1]))
        self._window_paint(event)
        gc.end()


def font_metrics_provider():
    gc = GraphicsContext((1, 1))
    gc.set_font(Font())
    return gc
