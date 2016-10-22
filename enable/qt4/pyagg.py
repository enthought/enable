# -----------------------------------------------------------------------------
# Copyright (c) 2016, Enthought, Inc.
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in enthought/LICENSE.txt and may be redistributed only
# under the conditions described in the aforementioned license.  The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
# Thanks for using Enthought open source!
# -----------------------------------------------------------------------------

from pyface.qt import QtCore, QtGui
from kiva.pyagg import CompiledPath, GraphicsContext  # noqa

from .base_window import BaseWindow
from .scrollbar import NativeScrollBar  # noqa


class Window(BaseWindow):
    def _create_gc(self, size, pix_format="rgba32"):
        gc = GraphicsContext((size[0]+1, size[1]+1), pix_format=pix_format)
        gc.translate_ctm(0.5, 0.5)

        return gc

    def _window_paint(self, event):
        if self.control is None:
            return

        # self._gc is an image context
        w = self._gc.width()
        h = self._gc.height()
        data = self._gc.gc.array
        image = QtGui.QImage(data, w, h, QtGui.QImage.Format_RGB32)
        rect = QtCore.QRect(0, 0, w, h)
        painter = QtGui.QPainter(self.control)
        painter.drawImage(rect, image)


def font_metrics_provider():
    from kiva.fonttools import Font
    gc = GraphicsContext((1, 1))
    gc.set_font(Font())
    return gc
