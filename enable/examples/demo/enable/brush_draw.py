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
This demonstrates using Brush to set the fill of a region.
"""
from traits.api import Instance

from enable.example_support import DemoFrame, demo_main
from enable.api import (
    Brush, ColorStop, Component, Container, Gradient, RadialGradientBrush
)


class Box(Component):

    resizable = ""

    brush = Instance(Brush)

    def _draw_mainlayer(self, gc, view_bounds=None, mode="default"):
        with gc:
            if self.brush is not None:
                self.brush.set_brush(gc)
            dx, dy = self.bounds
            x, y = self.position
            gc.rect(x, y, dx, dy)
            gc.fill_path()


class Demo(DemoFrame):
    def _create_component(self):
        box = Box(
            bounds=[100.0, 100.0],
            position=[50.0, 50.0],
            brush=RadialGradientBrush(
                center=(0.5, 0.5),
                radius=0.5,
                focus=(0.75, 0.75),
                gradient=Gradient(
                    stops=[
                        ColorStop(offset=0.0, color="red"),
                        ColorStop(offset=1.0, color="yellow"),
                    ],
                ),
                spread_method='pad',
                units="objectBoundingBox",
            )
        )
        container = Container(bounds=[500, 500])
        container.add(box)
        return container


if __name__ == "__main__":
    # Save demo so that it doesn't get garbage collected when run within
    # existing event loop (i.e. from ipython).
    demo = demo_main(Demo)
