# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
from kiva.api import FILL

# Relative imports
from .abstract_window import AbstractWindow
from .base import bounding_coordinates, coordinates_to_bounds
from .kiva_graphics_context import GraphicsContext


class EnableGCMixin(object):
    """
    Subclass of Kiva GraphicsContext that provides a few more utility methods.
    Most importantly, it provides a pointer back to the window that this
    GC is being drawn to.

    This will eventually be deprecated as the follow methods are folded into
    Kiva or their use is discontinuted in Enable.
    """

    # The window that this GraphicsContext is being drawn to.  It is OK to
    # leave this as None if the graphics context is used as a backbuffer;
    # however, in such cases, it is more appropriate to use a GC from Kiva
    # directly as opposed to using the Enable one, as some draw methods may
    # need to parent controls or dialogs from the Window.
    window = None  # Instance(AbstractWindow)

    def __init__(self, *args, **kwargs):
        if "window" in kwargs:
            self.window = kwargs.pop("window")
        super(EnableGCMixin, self).__init__(*args, **kwargs)

    def clip_to_rect(self, x, y, width, height):
        if getattr(self, "corner_pixel_origin", True):
            super(EnableGCMixin, self).clip_to_rect(
                x - 0.5, y - 0.5, width + 1, height + 1
            )
        else:
            super(EnableGCMixin, self).clip_to_rect(x, y, width, height)

    def clear_clip(self, color, coordinates):
        """ Clip and clear a Kiva graphics context to a specified area and
        color
        """
        bounds = coordinates_to_bounds(coordinates)
        self.clip_to_rect(*bounds)
        self.set_fill_color(color)
        self.draw_rect(bounds, FILL)

    def clear_clip_region(self, color, update_region):
        """ Clip and clear a Kiva graphics context to a specified region and
        color
        """
        bounds = coordinates_to_bounds(bounding_coordinates(update_region))
        self.clip_to_rect(*bounds)
        self.set_fill_color(color)
        for coordinates in update_region:
            bounds = coordinates_to_bounds(coordinates)
            self.begin_path()
            self.rect(*bounds)
        self.fill_path()

    def alpha(self, alpha):
        raise NotImplementedError("The alpha() method is not compatible with "
                                  "DisplayPDF; use clear() instead.")

    def stretch_draw(self, image, x, y, dx, dy):
        """ Draws an image 'stretched' to fit a specified area
        """
        idx = image.width()
        idy = image.height()
        with self:
            self.clip_to_rect(x, y, dx, dy)
            cx, cy, cdx, cdy = x, y, dx, dy
            yt = cy + cdy
            xr = cx + cdx
            x += (int(cx - x) / idx) * idx
            y += (int(cy - y) / idy) * idy
            while y < yt:
                x0 = x
                while x0 < xr:
                    self.draw_image(image, (x0, y, idx, idy))
                    x0 += idx
                y += idy


# Define a GraphicsContextEnable that subclasses whatever the Kiva backend's
# GraphicsContext is.
class GraphicsContextEnable(EnableGCMixin, GraphicsContext):
    pass


# Define an ImageGraphicsContextEnable that is guaranteed to be a subclass of
# an ImageGraphicsContext, regardless of the actual Kiva backend.  If the kiva
# backend is already the GraphicsContextImage, then just create an alias.
from kiva.image import GraphicsContext as GraphicsContextImage

if isinstance(GraphicsContext, GraphicsContextImage):
    ImageGraphicsContextEnable = GraphicsContextEnable
else:

    class ImageGraphicsContextEnable(EnableGCMixin, GraphicsContextImage):
        pass
