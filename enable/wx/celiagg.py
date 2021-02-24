# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!

import numpy as np
import wx

from kiva.celiagg import CompiledPath, GraphicsContext  # noqa

from .base_window import BaseWindow
from .scrollbar import NativeScrollBar


class Window(BaseWindow):
    def _create_gc(self, size, pix_format="rgba32"):
        gc = GraphicsContext(
            (size[0] + 1, size[1] + 1),
            pix_format=pix_format,
            base_pixel_scale=self.base_pixel_scale,
        )
        gc.translate_ctm(0.5, 0.5)
        return gc

    def _window_paint(self, event):
        if self.control is None:
            event.Skip()
            return

        control = self.control
        wdc = control._dc = wx.PaintDC(control)

        bmp = wx.Bitmap.FromBufferRGBA(
            self._gc.width(), self._gc.height(), np.ravel(self._gc.gc.array)
        )
        bmp.SetSize(control.GetSize())
        wdc.DrawBitmap(bmp, 0, 0)

        control._dc = None


def font_metrics_provider():
    from kiva.api import Font

    gc = GraphicsContext((1, 1))
    gc.set_font(Font())
    return gc
