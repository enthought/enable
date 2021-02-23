# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
from enable.api import Canvas, Viewport
from enable.example_support import demo_main, DemoFrame
from enable.tools.api import ViewportPanTool


class Demo(DemoFrame):
    def _create_component(self):
        canvas = Canvas(bgcolor="lightsteelblue", draw_axes=True)
        from enable.examples.demo.enable.basic_move import Box

        box = Box(color="red", bounds=[50, 50], resizable="")
        box.position = [75, 75]
        canvas.add(box)
        viewport = Viewport(component=canvas)
        viewport.view_position = [0, 0]
        viewport.tools.append(ViewportPanTool(viewport))
        return viewport


if __name__ == "__main__":
    # Save demo so that it doesn't get garbage collected when run within
    # existing event loop (i.e. from ipython).
    demo = demo_main(Demo, title="Canvas example")
