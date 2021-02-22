# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
from functools import reduce

import wx

from kiva.cairo import CompiledPath, GraphicsContext, font_metrics_provider

from .base_window import BaseWindow
from .scrollbar import NativeScrollBar

from enable.base import union_bounds


class Window(BaseWindow):
    def _create_gc(self, size, pix_format="bgra32"):
        "Create a Kiva graphics context of a specified size"
        gc = GraphicsContext(
            (size[0] + 1, size[1] + 1),
            base_pixel_scale=self.base_pixel_scale,
        )
        gc.translate_ctm(0.5, 0.5)
        return gc

    def _window_paint(self, event):
        "Do a GUI toolkit specific screen update"
        if self.control is None:
            event.Skip()
            return

        control = self.control
        pixel_map = self._gc.pixel_map
        control._dc = wx.PaintDC(control)
        self._update_region = None
        if self._update_region is not None:
            update_bounds = reduce(union_bounds, self._update_region)
            pixel_map.draw_to_wxwindow(
                control,
                int(update_bounds[0]),
                int(update_bounds[1]),
                width=int(update_bounds[2]),
                height=int(update_bounds[3]),
            )
        else:
            pixel_map.draw_to_wxwindow(control, 0, 0)

        control._dc = None
