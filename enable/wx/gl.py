# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
import wx

import pyglet

pyglet.options["shadow_window"] = False
from wx.glcanvas import GLCanvas

from traits.api import Instance
from kiva.gl import CompiledPath, FakePygletContext, GraphicsContext

from .base_window import BaseWindow
from .scrollbar import NativeScrollBar


class Window(BaseWindow):

    control = Instance(GLCanvas)

    def __init__(self, *args, **kw):
        super(Window, self).__init__(*args, **kw)

        # If we are using the GL backend, we will need to have a pyglet
        # GL context
        self._pyglet_gl_context = None

        self._gc = None

    def _create_control(self, parent, wid, pos=wx.DefaultPosition,
                        size=wx.DefaultSize):
        return GLCanvas(
            parent, wid, pos, size, style=wx.CLIP_CHILDREN | wx.WANTS_CHARS
        )

    def _create_gc(self, size, pix_format=None):
        """ Create a GraphicsContext instance.
        """
        gc = GraphicsContext(
            (size[0] + 1, size[1] + 1),
            base_pixel_scale=self.base_pixel_scale,
        )
        if self._pyglet_gl_context is None:
            self._pyglet_gl_context = FakePygletContext()
        gc.gl_init()
        gc.translate_ctm(0.5, 0.5)
        return gc

    def _init_gc(self):
        """ Gives the GC a chance to initialize itself before components
        perform layout and draw.  This is called every time through the paint
        loop.
        """
        dc = wx.PaintDC(self.control)
        self._pyglet_gl_context.set_current()
        self.control.SetCurrent()
        super(Window, self)._init_gc()

    def _paint(self, event=None):
        """ Paint the contents of the window.
        """
        if self.control is None:
            event.Skip()
            return

        size = self._get_control_size()
        self._size = tuple(size)
        self._gc = self._create_gc(size)
        self._init_gc()
        if hasattr(self.component, "do_layout"):
            self.component.do_layout()
        self._gc.clear(self.bgcolor_)
        self.component.draw(self._gc, view_bounds=(0, 0, size[0], size[1]))
        self._update_region = []
        self.control.SwapBuffers()


def font_metrics_provider():
    from kiva.api import Font

    gc = GraphicsContext((1, 1))
    gc.set_font(Font())
    return gc
