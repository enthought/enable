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
from kiva.celiagg import CompiledPath, GraphicsContext  # noqa
from pyface.qt import QtCore, QtGui
from traits.api import Array

from .base_window import BaseWindow
from .scrollbar import NativeScrollBar  # noqa


class Window(BaseWindow):
    # Keep a buffer around for converting RGBA -> BGRA
    _shuffle_buffer = Array(shape=(None, None, 4), dtype=np.uint8)

    def _create_gc(self, size, pix_format="rgba32"):
        gc = GraphicsContext(
            (size[0] + 1, size[1] + 1),
            pix_format=pix_format,
            base_pixel_scale=self.base_pixel_scale,
        )
        gc.translate_ctm(0.5, 0.5)

        self._shuffle_buffer = np.empty(
            (size[1] + 1, size[0] + 1, 4), dtype=np.uint8
        )

        return gc

    def _window_paint(self, event):
        if self.control is None:
            return

        # Convert to Qt's pixel format
        self._shuffle_copy()

        # self._gc is an image context
        w = self._gc.width()
        h = self._gc.height()
        image = QtGui.QImage(
            self._shuffle_buffer, w, h, QtGui.QImage.Format_RGB32
        )
        rect = QtCore.QRectF(0, 0, self.control.width(), self.control.height())
        painter = QtGui.QPainter(self.control)
        painter.drawImage(rect, image)

    def _shuffle_copy(self):
        """ Convert from RGBA -> BGRA.
        Supported source formats are: rgb24, rgba32, & bgra32

        Qt's Format_RGB32 is actually BGR. So, Yeah...
        """
        src = self._gc.gc.array
        dst = self._shuffle_buffer
        src_fmt = self._gc.pix_format

        if src_fmt.startswith("rgb"):
            indices = (2, 1, 0)
        else:
            indices = (0, 1, 2)
        dst[..., 0] = src[..., indices[0]]
        dst[..., 1] = src[..., indices[1]]
        dst[..., 2] = src[..., indices[2]]

        if src_fmt in ("rgba32", "bgra32"):
            dst[..., 3] = src[..., 3]
        else:
            dst[..., 3] = 255


def font_metrics_provider():
    from kiva.api import Font

    gc = GraphicsContext((1, 1))
    gc.set_font(Font())
    return gc
