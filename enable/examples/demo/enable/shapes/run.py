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
An example showing moveable shapes.
"""

# Enthought library imports.
from enable.api import Container
from enable.example_support import DemoFrame, demo_main

# Local imports
from enable.examples.demo.enable.shapes.api import Box, Circle


class Demo(DemoFrame):
    """ The top-level frame. """

    # 'DemoFrame' interface.
    # --------------------------------------------------------------------------

    def _create_component(self):

        container = Container(
            auto_size=False, bgcolor="black", *self._create_shapes()
        )

        return container

    # Private interface.
    # --------------------------------------------------------------------------

    def _create_shapes(self):
        """ Create some shapes. """

        box1 = Box(
            bounds=[100, 100],
            position=[50, 50],
            fill_color="lightpink",
            text="Box 1",
        )

        box2 = Box(
            bounds=[100, 100],
            position=[150, 150],
            fill_color="greenyellow",
            text="Box 2",
        )

        circle1 = Circle(
            radius=50,
            position=[250, 250],
            fill_color="cornflowerblue",
            text="Circle 1",
        )

        circle2 = Circle(
            radius=50, position=[350, 350], fill_color="khaki", text="Circle 2"
        )

        return box1, box2, circle1, circle2


if __name__ == "__main__":
    # Save demo so that it doesn't get garbage collected when run within
    # existing event loop (i.e. from ipython).
    demo = demo_main(Demo, size=(500, 500), title="Click and drag the shapes")
