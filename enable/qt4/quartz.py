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

from kiva.api import Font
from kiva.quartz import get_mac_context, ABCGI

from .base_window import BaseWindow
from .scrollbar import NativeScrollBar

CompiledPath = ABCGI.CGMutablePath


class GraphicsContext(ABCGI.CGLayerContext):
    def __init__(self, size_or_array, *args, **kwds):
        gc = kwds.pop("window_gc", None)
        if not gc:
            # Create a tiny base context to spawn the CGLayerContext from.
            # We are better off making our Layer from the window gc since
            # the data formats will match and so it will be faster to draw the
            # layer.
            gc = ABCGI.CGBitmapContext((1, 1))
        if isinstance(size_or_array, np.ndarray):
            # Initialize the layer with an image.
            image = ABCGI.CGImage(size_or_array)
            width = image.width
            height = image.height
        else:
            # No initialization.
            image = None
            width, height = size_or_array

        super(GraphicsContext, self).__init__(
            (width, height), gc, *args, **kwds
        )
        if image is not None:
            self.draw_image(image)

    @classmethod
    def create_from_gc(klass, gc, size_or_array, *args, **kwds):
        return klass(size_or_array, window_gc=gc, *args, **kwds)


class _WindowGraphicsContext(ABCGI.CGContextInABox):
    def __init__(self, *args, **kwds):
        super(_WindowGraphicsContext, self).__init__(*args, **kwds)
        self._begun = False

    def begin(self):
        if self._begun:
            return
        self.save_state()
        self.translate_ctm(0, self.height())
        self.scale_ctm(1.0, -1.0)
        self._begun = True

    def end(self):
        if self._begun:
            self.restore_state()
            self._begun = False

    @staticmethod
    def create_from_gc(gc, size_or_array, *args, **kwds):
        return GraphicsContext(size_or_array, window_gc=gc, *args, **kwds)


class Window(BaseWindow):
    """ An Enable Window for Qt GUIs on OS X.
    """

    # 'BaseWindow' interface ###############################################

    def _create_gc(self, size, pix_format=None):
        if hasattr(self.control, "winId"):
            # From the Qt 4.7 Documentation:
            # "On Mac OS X, the type returned depends on which framework Qt was
            # linked against. If Qt is using Carbon, the {WId} is actually
            # an HIViewRef. If Qt is using Cocoa, {WId} is a pointer to
            # an NSView."
            # get_mac_context() only works on Cocoa.
            self.dc = get_mac_context(self.control.winId())
            if self.dc:
                gc = _WindowGraphicsContext(size, self.dc)
                gc.begin()
                return gc
        raise NotImplementedError("Only Qt built against Cocoa is supported")

    def _window_paint(self, event):
        # Make sure end() is called so that the window's GC is not left in an
        # odd state.
        self._gc.end()
        # Force a new gc to be created for the next paint()
        self._gc = None
        self.dc = None


def font_metrics_provider():
    gc = GraphicsContext((1, 1))
    gc.set_font(Font())
    return gc
