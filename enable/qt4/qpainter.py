# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!

from kiva.qpainter import CompiledPath, GraphicsContext, font_metrics_provider

from .base_window import BaseWindow
from .scrollbar import NativeScrollBar


class Window(BaseWindow):
    def _create_gc(self, size, pix_format=None):
        gc = GraphicsContext(
            (size[0] + 1, size[1] + 1),
            base_pixel_scale=self.base_pixel_scale,
            parent=self.control,
        )
        gc.translate_ctm(0.5, 0.5)

        return gc

    def _window_paint(self, event):
        # get rid of the GC after drawing
        self._gc = None
