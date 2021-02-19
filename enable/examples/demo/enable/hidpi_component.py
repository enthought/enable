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
Basic demo of controlling the pixel resolution of an Enable component.

HiDPI graphics are available when using Qt 5.x+ or WxWidgets 4.x+. With Qt 4.x,
you will not see any difference between the two components.
"""
import math

from traits.api import Any, HasTraits, Instance, Str
from traitsui.api import HGroup, UItem, View

from enable.api import Component, ComponentEditor, str_to_font


class MyComponent(Component):
    quote = Str()
    _font = Any()

    def draw(self, gc, **kwargs):
        if not self._font:
            self._font = str_to_font(None, None, "modern 48")

        gc.clear((0.5, 0.5, 0.5))
        mx = self.x + self.width / 2.0
        my = self.y + self.height / 2.0
        with gc:
            gc.set_fill_color((1.0, 1.0, 0.0, 1.0))
            gc.arc(mx, my, 100, 0, 2 * math.pi)
            gc.fill_path()

            gc.set_font(self._font)
            tx, ty, tw, th = gc.get_text_extent(self.quote)
            tx = mx - tw / 2.0
            ty = my - th / 2.0
            gc.set_fill_color((0.0, 0.0, 0.0, 1.0))
            gc.show_text_at_point(self.quote, tx, ty)


class Demo(HasTraits):
    lodpi = Instance(Component)
    hidpi = Instance(Component)

    traits_view = View(
        HGroup(
            HGroup(
                UItem(
                    "lodpi",
                    editor=ComponentEditor(high_resolution=False),
                    width=250,
                    height=250,
                ),
            ),
            HGroup(
                UItem(
                    "hidpi",
                    editor=ComponentEditor(high_resolution=True),
                    width=250,
                    height=250,
                ),
            ),
        ),
        resizable=True,
        title="HiDPI Example",
    )

    def _lodpi_default(self):
        return MyComponent(quote="Pixelated")

    def _hidpi_default(self):
        return MyComponent(quote="Smooth")


if __name__ == "__main__":
    Demo().configure_traits()
