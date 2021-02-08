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
Similar to simple_drag_demo, put one circle inside a scrolled container
"""
from numpy import array
from traits.api import Enum, Float, Instance, Tuple

from enable.api import Component, Scrolled, Container, Pointer
from enable.example_support import DemoFrame, demo_main


class Circle(Component):
    """
    The circle moves with the mouse cursor but leaves a translucent version of
    itself in its original position until the mouse button is released.
    """

    color = (0.3, 0.4, 0.8, 1.0)
    bgcolor = "none"

    normal_pointer = Pointer("arrow")
    moving_pointer = Pointer("hand")

    offset_x = Float
    offset_y = Float

    shadow_type = Enum("light", "dashed")
    shadow = Instance(Component)

    def __init__(self, **traits):
        Component.__init__(self, **traits)
        self.pointer = self.normal_pointer

    def _draw_mainlayer(self, gc, view_bounds=None, mode="default"):
        with gc:
            gc.set_fill_color(self.color)
            dx, dy = self.bounds
            x, y = self.position
            radius = min(dx / 2.0, dy / 2.0)
            gc.arc(x + dx / 2.0, y + dy / 2.0, radius, 0.0, 2 * 3.14159)
            gc.fill_path()

    def normal_left_down(self, event):
        self.event_state = "moving"
        self.pointer = self.moving_pointer

        # Create our shadow
        if self.shadow_type == "light":
            klass = LightCircle
        else:
            klass = DashedCircle
        dx, dy = self.bounds
        self.shadow = klass(
            bounds=self.bounds, position=self.position, color=self.color
        )
        self.container.add(self.shadow)
        x, y = self.position
        self.offset_x = event.x - x
        self.offset_y = event.y - y

    def moving_mouse_move(self, event):
        self.position = [event.x - self.offset_x, event.y - self.offset_y]
        self.request_redraw()

    def moving_left_up(self, event):
        self.event_state = "normal"
        self.pointer = self.normal_pointer
        self.request_redraw()
        # Remove our shadow
        self.container.remove(self.shadow)

    def moving_mouse_leave(self, event):
        self.moving_left_up(event)


class LightCircle(Component):

    color = Tuple
    bgcolor = "none"
    radius = Float(1.0)

    def _draw_mainlayer(self, gc, view_bounds=None, mode="default"):
        with gc:
            gc.set_fill_color(self.color[0:3] + (self.color[3] * 0.3,))
            dx, dy = self.bounds
            x, y = self.position
            radius = min(dx / 2.0, dy / 2.0)
            gc.arc(x + dx / 2.0, y + dy / 2.0, radius, 0.0, 2 * 3.14159)
            gc.fill_path()


class DashedCircle(Component):

    color = Tuple
    bgcolor = "none"
    radius = Float(1.0)
    line_dash = array([2.0, 2.0])

    def _draw_mainlayer(self, gc, view_bounds=None, mode="default"):
        with gc:
            gc.set_fill_color(self.color)
            dx, dy = self.bounds
            x, y = self.position
            radius = min(dx / 2.0, dy / 2.0)
            gc.arc(x + dx / 2.0, y + dy / 2.0, radius, 0.0, 2 * 3.14159)
            gc.set_stroke_color(self.color[0:3] + (self.color[3] * 0.8,))
            gc.set_line_dash(self.line_dash)
            gc.stroke_path()


class Demo(DemoFrame):
    def _create_component(self):
        container = Container(
            bounds=[800, 600],
            bgcolor=(0.9, 0.7, 0.7, 1.0),
            auto_size=False,
            fit_window=False,
        )
        circle1 = Circle(
            bounds=[75, 75], position=[100, 100], shadow_type="dashed"
        )
        container.add(circle1)

        scr = Scrolled(
            container,
            bounds=[200, 200],
            position=[50, 50],
            stay_inside=True,
            vertical_anchor="top",
            horizontal_anchor="left",
            fit_window=False,
        )
        return scr


if __name__ == "__main__":
    # Save demo so that it doesn't get garbage collected when run within
    # existing event loop (i.e. from ipython).
    demo = demo_main(Demo, title="Click and drag to move the circles")
