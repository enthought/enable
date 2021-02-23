# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
""" Defines the Image component class.
"""

# Enthought library imports
from traits.api import Array, Enum, Property, cached_property

# Local imports
from enable.component import Component


class Image(Component):
    """ Component that displays a static image

    This is extremely simple right now.  By default it will draw the array into
    the entire region occupied by the component, stretching or shrinking as
    needed.  By default the bounds are set to the width and height of the data
    array, and we provide the same information to constraints-based layout
    with the layout_size_hint trait.
    """

    #: the image data as an array
    data = Array(shape=(None, None, (3, 4)), dtype="uint8")

    #: the format of the image data (eg. RGB vs. RGBA)
    format = Property(Enum("rgb24", "rgba32"), depends_on="data")

    #: the size-hint for constraints-based layout
    layout_size_hint = Property(data, depends_on="data")

    #: the image as a C-contiguous ndarray
    _image = Property(Array(shape=(None, None, None)), depends_on="data")

    @classmethod
    def from_file(cls, filename, **traits):
        from PIL import Image
        from numpy import asarray

        data = asarray(Image.open(filename))
        return cls(data=data, **traits)

    def __init__(self, data, **traits):
        # the default bounds are the size of the image
        traits.setdefault("bounds", data.shape[1::-1])
        super(Image, self).__init__(data=data, **traits)

    def _draw_mainlayer(self, gc, view_bounds=None, mode="normal"):
        """ Draws the image. """
        with gc:
            rect = (self.x, self.y, self.width, self.height)
            gc.draw_image(self._image, rect)

    @cached_property
    def _get_format(self):
        if self.data.shape[-1] == 3:
            return "rgb24"
        elif self.data.shape[-1] == 4:
            return "rgba32"
        else:
            raise ValueError("Data array not correct shape")

    @cached_property
    def _get_layout_size_hint(self):
        return self.data.shape[1::-1]

    @cached_property
    def _get__image(self):
        if not self.data.flags["C_CONTIGUOUS"]:
            data = self.data.copy()
        else:
            data = self.data
        return data
