# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
from PIL import Image, ImageQt

# Qt imports.
from pyface.qt import QtCore, QtGui

from kiva.api import Font
from kiva.quartz import ABCGI

from .base_window import BaseWindow
from .scrollbar import NativeScrollBar

CompiledPath = ABCGI.CGMutablePath


class GraphicsContext(ABCGI.CGBitmapContext):
    pass


class Window(BaseWindow):
    """ An Enable Window for Qt GUIs on OS X.
    """

    # 'BaseWindow' interface ###############################################

    def _create_gc(self, size, pix_format=None):
        gc = GraphicsContext(size)
        gc.scale_ctm(self.base_pixel_scale, self.base_pixel_scale)

        return gc

    def _window_paint(self, event):
        if self.control is None:
            return

        # self._gc is an image context
        w = self._gc.width()
        h = self._gc.height()
        pilimage = Image.frombuffer("RGBA", (w, h), self._gc,
                                    "raw", "RGBA", 0, 1)
        image = ImageQt.toqimage(pilimage)
        rect = QtCore.QRectF(0, 0, self.control.width(), self.control.height())
        painter = QtGui.QPainter(self.control)
        painter.drawImage(rect, image)


def font_metrics_provider():
    gc = GraphicsContext((1, 1))
    gc.set_font(Font())
    return gc
