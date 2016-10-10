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

import pyglet
pyglet.options['shadow_window'] = False

from traits.api import Bool, Instance
from kiva.gl import CompiledPath, GraphicsContext

from .base_window import BaseGLWindow
from .scrollbar import NativeScrollBar

class Window(BaseGLWindow):
    def _create_gc(self, size, pix_format=None):
        """ Create a GraphicsContext instance.
        """
        from pyglet.gl import Context

        gc = GraphicsContext((size[0]+1, size[1]+1))
        self._pyglet_gl_context = Context()
        gc.gl_init()
        gc.translate_ctm(0.5, 0.5)
        return gc

    def _init_gc(self):
        """ Gives the GC a chance to initialize itself before components perform layout
        and draw.  This is called every time through the paint loop.
        """
        self._pyglet_gl_context.set_current()
        self.control.makeCurrent()
        super(Window, self)._init_gc()

    def _paint(self, event=None):
        """ Paint the contents of the window.
        """
        if self.control is None:
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

def font_metrics_provider():
    from kiva.fonttools import Font
    gc = GraphicsContext((1, 1))
    gc.set_font(Font())
    return gc

# EOF
