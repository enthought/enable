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
Button Tool
===========
"""

from traits.api import Bool
from kiva.api import FILL, STROKE

from enable.api import Container, transparent_color
from enable.colors import ColorTrait
from enable.example_support import DemoFrame, demo_main
from enable.primitives.api import Box
from enable.tools.button_tool import ButtonTool


class SelectableBox(Box):
    """ A box that can be selected and renders in a different color """

    selected = Bool

    selected_color = ColorTrait("green")

    def select(self, selected):
        self.selected = selected

    def _selected_changed(self):
        self.request_redraw()

    def _draw_mainlayer(self, gc, view_bounds=None, mode="default"):
        "Draw the box background in a specified graphics context"

        # Set up all the control variables for quick access:
        bs = self.border_size
        bsd = bs + bs
        bsh = bs / 2.0
        x, y = self.position
        dx, dy = self.bounds

        with gc:
            if self.selected:
                color = self.selected_color_
            else:
                color = self.color_
            if color is not transparent_color:
                gc.set_fill_color(color)
                gc.draw_rect((x + bs, y + bs, dx - bsd, dy - bsd), FILL)

            # Draw the border (if required):
            if bs > 0:
                border_color = self.border_color_
                if border_color is not transparent_color:
                    gc.set_stroke_color(border_color)
                    gc.set_line_width(bs)
                    gc.draw_rect((x + bsh, y + bsh, dx - bs, dy - bs), STROKE)


class MyFrame(DemoFrame):
    """ Example of using a ButtonTool

    """

    def button_clicked(self):
        print("clicked")

    # -------------------------------------------------------------------------
    # DemoApplication interface
    # -------------------------------------------------------------------------

    def _create_component(self):
        # create a box that changes color when clicked
        push_button_box = SelectableBox(
            bounds=[100, 50], position=[50, 50], color="red"
        )

        # create a basic push button tool for it
        push_button_tool = ButtonTool(component=push_button_box)
        push_button_box.tools.append(push_button_tool)

        # print when box clicked, change color when button down
        push_button_tool.on_trait_change(self.button_clicked, "clicked")
        push_button_tool.on_trait_change(push_button_box.select, "down")

        # another box for a toggle button
        toggle_box = SelectableBox(
            bounds=[100, 50],
            position=[50, 125],
            color="lightblue",
            selected_color="blue",
        )

        # a toggle button tool
        toggle_button_tool = ButtonTool(component=toggle_box, togglable=True)
        toggle_box.tools.append(toggle_button_tool)

        # change color when checked down
        toggle_button_tool.on_trait_change(toggle_box.select, "checked")

        container = Container(bounds=[600, 600])
        container.add(push_button_box)
        container.add(toggle_box)

        return container


if __name__ == "__main__":
    # Save demo so that it doesn't get garbage collected when run within
    # existing event loop (i.e. from ipython).
    demo = demo_main(MyFrame, size=(600, 600))
