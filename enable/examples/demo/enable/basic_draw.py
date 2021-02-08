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
This demonstrates the most basic drawing capabilities using Enable.  A new
component is created and added to a container.
"""
from enable.example_support import DemoFrame, demo_main
from enable.api import Component, Container


class Box(Component):

    resizable = ""

    def _draw_mainlayer(self, gc, view_bounds=None, mode="default"):
        with gc:
            gc.set_fill_color((1.0, 0.0, 0.0, 1.0))
            dx, dy = self.bounds
            x, y = self.position
            gc.rect(x, y, dx, dy)
            gc.fill_path()


class Demo(DemoFrame):
    def _create_component(self):
        box = Box(bounds=[100.0, 100.0], position=[50.0, 50.0])
        container = Container(bounds=[500, 500])
        container.add(box)
        return container


if __name__ == "__main__":
    # Save demo so that it doesn't get garbage collected when run within
    # existing event loop (i.e. from ipython).
    demo = demo_main(Demo)
