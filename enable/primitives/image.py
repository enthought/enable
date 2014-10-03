""" Defines the Label class.
"""

from __future__ import absolute_import

# Enthought library imports
from traits.api import Array, Bool, Enum, Instance, Property, cached_property

# Local imports
from enable.component import Component
from kiva.image import GraphicsContext


class Image(Component):
    """ Component that displays a static image

    This is extremely simple right now.  By default it will draw the array into
    the entire region occupied by the component, stretching or shrinking as
    needed.  It is up to the user to set the bounds, although the from_file
    will give something sensible, and we assume that a constraints-based layout
    should have a layour_size_hint of the image size.

    """

    #: the image data as an array
    data = Array(dtype='uint8')

    #: the format of the image data (eg. RGB vs. RGBA)
    format = Property(Enum('rgb24', 'rgba32'))

    #: the size-hint for constraints-based layout
    layout_size_hint = Property(data)

    #: the image as an Image GC
    _image = Property(Instance(GraphicsContext), depends_on='data')

    @classmethod
    def from_file(cls, filename):
        from scipy.misc import imread
        data = imread(filename)
        return cls(data=data, bounds=data.shape[:2])

    def _draw_mainlayer(self, gc, view_bounds=None, mode="normal"):
        """ Draws the image. """
        with gc:
            gc.draw_image(self._image, (self.x, self.y, self.width, self.height))

    @cached_property
    def _get_format(self):
        if self.data.shape[-1] == 3:
            return 'rgb24'
        elif self.data.shape[-1] == 4:
            return 'rgba32'
        else:
            raise ValueError('Data array not correct shape')

    @cached_property
    def _get_layout_size_hint(self):
        return self.data.shape[:2]

    @cached_property
    def _get__image(self):
        if not self.data.flags['C_CONTIGUOUS']:
            data = self.data.copy()
        else:
            data = self.data
        image_gc = GraphicsContext(data, pix_format=self.format)
        return image_gc
