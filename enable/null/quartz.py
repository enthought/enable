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
from kiva.quartz import ABCGI


class NativeScrollBar(object):
    pass


class Window(object):
    pass


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
        return klass(size_or_array, gc, *args, **kwds)


def font_metrics_provider():
    gc = GraphicsContext((1, 1))
    gc.set_font(Font())
    return gc
