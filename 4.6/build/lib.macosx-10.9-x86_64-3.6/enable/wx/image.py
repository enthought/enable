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
import sys
import wx

from kiva.agg import CompiledPath, GraphicsContextSystem as GraphicsContext

from .base_window import BaseWindow
from .scrollbar import NativeScrollBar


def _wx_bitmap_from_buffer(buf, width, height):
    """ Given a pixel buffer in ARGB order, return a WX bitmap
        object with the pixels in BGRA order.
    """
    arr = np.frombuffer(buf, dtype=np.uint8).reshape((width, height, 4))
    copy = np.zeros_like(arr)
    copy[..., 0::4] = arr[..., 2::4]
    copy[..., 1::4] = arr[..., 1::4]
    copy[..., 2::4] = arr[..., 0::4]
    copy[..., 3::4] = arr[..., 3::4]
    return wx.Bitmap.FromBufferRGBA(width, height, np.ravel(copy))


class Window(BaseWindow):
    def _create_gc(self, size, pix_format="bgra32"):
        "Create a Kiva graphics context of a specified size"
        # We have to set bottom_up=0 or otherwise the PixelMap will
        # appear upside down when blitting. Note that this is not the
        # case on Windows.
        bottom_up = 0 if sys.platform != "win32" else 1
        gc = GraphicsContext(
            (size[0] + 1, size[1] + 1),
            pix_format=pix_format,
            base_pixel_scale=self.base_pixel_scale,
            bottom_up=bottom_up,
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
        wdc = control._dc = wx.PaintDC(control)

        if hasattr(pixel_map, "draw_to_wxwindow"):
            pixel_map.draw_to_wxwindow(control, 0, 0)
        else:
            # This should just be the Mac OS X code path
            bmp = _wx_bitmap_from_buffer(
                pixel_map.convert_to_argb32string(),
                self._gc.width(),
                self._gc.height(),
            )
            bmp.SetSize(control.GetSize())
            wdc.DrawBitmap(bmp, 0, 0)

        control._dc = None


def font_metrics_provider():
    from kiva.api import Font

    gc = GraphicsContext((1, 1))
    gc.set_font(Font())
    return gc
