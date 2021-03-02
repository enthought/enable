# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
import pyglet
from pyglet.image.codecs.png import PNGImageEncoder

import kiva.gl as gl_backend

# Pass it along
CompiledPath = gl_backend.CompiledPath


class GraphicsContext(gl_backend.GraphicsContext):
    """ This is a wrapper of the GL GraphicsContext which works in headless
    mode.
    """
    def __init__(self, size, *args, **kw):
        width, height = size
        self.__window = pyglet.window.Window(width=width, height=height)
        super().__init__((width, height), base_pixel_scale=1.0)
        self.gl_init()

    def clear(self, *args):
        self.__window.clear()

    def save(self, filename, *args, **kw):
        buffer = pyglet.image.get_buffer_manager()
        with open(filename, mode="wb") as fp:
            buffer.get_color_buffer().save(
                filename,
                file=fp,
                encoder=PNGImageEncoder(),
            )
