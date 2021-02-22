# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
"""
Image Explorer
==============

Interactive editor for exploring Kiva image drawing.
"""
import numpy as np
from PIL import Image

from traits.api import Enum, Instance
from traitsui.api import HSplit, Item, ModelView, VGroup, View

from enable.api import Component, ComponentEditor


class ImageComponent(Component):
    """ An Enable component that draws an image
    """
    #: The image data to draw
    image = Instance(Image.Image)

    #: What's the color space of the image?
    image_mode = Enum('RGB', 'RGBA', 'L', 'P')

    def _draw_mainlayer(self, gc, view_bounds=None, mode="default"):
        """ Try running the compiled code with the graphics context as `gc`
        """
        if self.image is None:
            return

        gc.draw_image(self.image, (0, 0, gc.width(), gc.height()))

    def _image_default(self):
        mode = self.image_mode
        components = 4 if mode == 'RGBA' else 3
        img = np.zeros((512, 512, components), dtype=np.uint8)

        RED, GREEN, BLUE = 0, 1, 2
        # Red, Green, Blue, and Cyan quadrants
        img[0:256, 0:256, RED] = 255
        img[0:256, 256:512, GREEN] = 255
        img[256:512, 0:256, BLUE] = 255
        img[256:512, 256:512, [GREEN, BLUE]] = 255

        if components == 4:
            img[:, :, 3] = np.linspace(0, 255, num=512*512).reshape(512, 512)

        return Image.fromarray(img).convert(mode)

    def _image_mode_changed(self):
        self.image = self._image_default()
        self.request_redraw()


class ImageComponentView(ModelView):
    """ ModelView of a ScriptedComponent displaying the script and image
    """

    #: the component we are editing
    model = Instance(ImageComponent, ())

    view = View(
        HSplit(
            VGroup(
                Item("model.image_mode", label="Colorspace"),
            ),
            VGroup(
                Item(
                    "model",
                    editor=ComponentEditor(),
                    springy=True,
                    show_label=False,
                ),
            ),
            show_border=True,
        ),
        resizable=True,
        title="Image Explorer",
    )


# "popup" is a magically named variable for the etsdemo application which will
# cause this demo to be run as a popup rather than trying to compress it into
# a tab on the application
popup = ImageComponentView()

if __name__ == "__main__":
    popup.configure_traits()
