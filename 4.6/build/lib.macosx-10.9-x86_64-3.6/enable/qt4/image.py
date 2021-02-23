# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!

from pyface.qt import QtCore, QtGui
from kiva.agg import CompiledPath, GraphicsContextSystem as GraphicsContext

from .base_window import BaseWindow
from .scrollbar import NativeScrollBar


class Window(BaseWindow):
    def _create_gc(self, size, pix_format="bgra32"):
        gc = GraphicsContext(
            (size[0] + 1, size[1] + 1),
            pix_format=pix_format,
            base_pixel_scale=self.base_pixel_scale,
            # We have to set bottom_up=0 or otherwise the PixelMap will
            # appear upside down in the QImage.
            bottom_up=0,
        )
        gc.translate_ctm(0.5, 0.5)

        return gc

    def _window_paint(self, event):
        if self.control is None:
            return

        # self._gc is an image context
        w = self._gc.width()
        h = self._gc.height()
        data = self._gc.pixel_map.convert_to_argb32string()
        image = QtGui.QImage(data, w, h, QtGui.QImage.Format_ARGB32)
        rect = QtCore.QRectF(0, 0, self.control.width(), self.control.height())
        painter = QtGui.QPainter(self.control)
        painter.drawImage(rect, image)


def font_metrics_provider():
    from kiva.api import Font

    gc = GraphicsContext((1, 1))
    gc.set_font(Font())
    return gc
