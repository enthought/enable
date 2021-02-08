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
**WARNING**

  This demo might not work as expected and some documented features might be
  missing.

"""
# Issue related to the demo warning: enthought/enable#501

from numpy import array

from enable.api import Canvas, Viewport, Scrolled
from enable.example_support import demo_main, DemoFrame
from enable.primitives.api import Box
from enable.tools.api import ViewportPanTool


class Demo(DemoFrame):
    def _create_component(self):
        canvas = Canvas(bgcolor="lightsteelblue", draw_axes=True)

        boxgridsize = 8
        boxsize = 50

        spacing = boxsize * 2
        offset = spacing / 2

        origin_color = array([0.0, 0.0, 1.0])
        x_color = array([0.0, 1.0, 0.0])
        y_color = array([1.0, 0.0, 0.0])

        for i in range(boxgridsize):
            for j in range(boxgridsize):
                color = tuple(
                    x_color / (boxgridsize - 1) * i
                    + y_color / (boxgridsize - 1) * j
                    + origin_color
                ) + (1.0,)
                box = Box(color=color, bounds=[boxsize, boxsize], resizable="")
                box.position = [
                    i * spacing + offset - boxsize / 2 + 0.5,
                    j * spacing + offset - boxsize / 2 + 0.5,
                ]
                canvas.add(box)

        viewport = Viewport(
            component=canvas,
            enable_zoom=True,
            vertical_anchor="center",
            horizontal_anchor="center",
        )
        # viewport.view_position = [0,0]
        viewport.tools.append(ViewportPanTool(viewport))

        # Uncomment the following to enforce limits on the zoom
        # viewport.min_zoom = 0.1
        # viewport.max_zoom = 3.0

        scrolled = Scrolled(
            canvas,
            fit_window=True,
            inside_padding_width=0,
            mousewheel_scroll=False,
            viewport_component=viewport,
            always_show_sb=True,
            continuous_drag_update=True,
        )

        return scrolled


if __name__ == "__main__":
    demo = demo_main(Demo, title="Canvas example")
