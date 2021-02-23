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
Shows how to implement a simple drag operation with transitory visual
side-effects that get cleaned up when the drag completes.  There is a
filled container that contains two filled circles, each of which has a
different kind of "shadow" object that it dynamically adds to the
container when the user clicks and drags the circle around.  Because
the shadow objects are "first-class" components in the filled container,
and because containers default to auto-sizing around their components,
the container stretches to the minimum bounding box of its components
as the user drags the circles around.
"""
from numpy import array
from traits.api import Any, Enum, Float, Instance, Tuple

from enable.api import Container, Component, Pointer, str_to_font
from enable.example_support import DemoFrame, demo_main


class MyFilledContainer(Container):

    fit_window = False
    border_width = 2
    resizable = ""
    _font = Any

    def _draw_container_mainlayer(self, gc, view_bounds, mode="default"):
        'Draws a filled container with the word "Container" in the center'
        if not self._font:
            self._font = str_to_font(None, None, "modern 10")

        with gc:
            gc.set_fill_color(self.bgcolor_)
            gc.rect(self.x, self.y, self.width, self.height)
            gc.draw_path()
            self._draw_border(gc)
            gc.set_font(self._font)
            gc.set_fill_color((1.0, 1.0, 1.0, 1.0))
            tx, ty, tw, th = gc.get_text_extent("Container")
            tx = self.x + self.width / 2.0 - tw / 2.0
            ty = self.y + self.height / 2.0 - th / 2.0
            gc.show_text_at_point("Container", tx, ty)


class Circle(Component):
    """
    The circle moves with the mouse cursor but leaves a translucent version of
    itself in its original position until the mouse button is released.
    """

    color = (0.6, 0.7, 1.0, 1.0)
    bgcolor = "none"

    normal_pointer = Pointer("arrow")
    moving_pointer = Pointer("hand")

    prev_x = Float
    prev_y = Float

    shadow_type = Enum("light", "dashed")
    shadow = Instance(Component)

    resizable = ""

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
        event.window.set_mouse_owner(self, event.net_transform())

        # Create our shadow
        if self.shadow_type == "light":
            klass = LightCircle
        else:
            klass = DashedCircle
        self.shadow = klass(
            bounds=self.bounds,
            position=self.position,
            color=(1.0, 1.0, 1.0, 1.0),
        )
        self.container.insert(0, self.shadow)
        x, y = self.position
        self.prev_x = event.x
        self.prev_y = event.y

    def moving_mouse_move(self, event):
        self.position = [
            self.x + (event.x - self.prev_x),
            self.y + (event.y - self.prev_y),
        ]
        self.prev_x = event.x
        self.prev_y = event.y
        self.request_redraw()

    def moving_left_up(self, event):
        self.event_state = "normal"
        self.pointer = self.normal_pointer
        event.window.set_mouse_owner(None)
        event.window.redraw()
        # Remove our shadow
        self.container.remove(self.shadow)

    def moving_mouse_leave(self, event):
        self.moving_left_up(event)


class LightCircle(Component):

    color = Tuple
    bgcolor = "none"
    radius = Float(1.0)
    resizable = ""

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
    resizable = ""

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
        circle1 = Circle(
            bounds=[75, 75], position=[50, 50], shadow_type="dashed"
        )
        circle2 = Circle(
            bounds=[75, 75], position=[200, 50], shadow_type="light"
        )
        container = MyFilledContainer(
            bounds=[500, 500], bgcolor=(0.5, 0.5, 0.5, 1.0)
        )
        container.auto_size = True
        container.add(circle1)
        container.add(circle2)
        return container


if __name__ == "__main__":
    # Save demo so that it doesn't get garbage collected when run within
    # existing event loop (i.e. from ipython).
    demo = demo_main(Demo, title="Click and drag to move the circles")
