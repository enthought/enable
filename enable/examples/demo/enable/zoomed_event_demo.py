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
Use mouse wheel to zoom and right-click to pan the viewport.
"""
from traits.api import Float

from enable.api import AbstractOverlay, Canvas, Viewport, ColorTrait, Scrolled
from enable.example_support import demo_main, DemoFrame
from enable.primitives.api import Box
from enable.tools.api import ViewportPanTool


class DropCanvas(Canvas):
    """ Adds a Box at a drop location """

    def normal_drag_over(self, event):
        self.window.set_drag_result("link")
        event.handled = True

    def normal_dropped_on(self, event):
        self.window.set_drag_result("link")
        print(event.obj)

        box = Box(x=event.x - 2, y=event.y - 2, width=4, height=4)
        self.add(box)

        self.request_redraw()
        event.handled = True


class EventTracer(AbstractOverlay):
    """ Draws a marker under the mouse cursor where an event is occurring. """

    x = Float
    y = Float

    color = ColorTrait("red")
    size = Float(5)
    angle = Float(0.0)  # angle in degrees

    def normal_mouse_move(self, event):
        self.x = event.x
        self.y = event.y
        self.component.request_redraw()

    def overlay(self, component, gc, view_bounds, mode):
        with gc:
            gc.translate_ctm(self.x, self.y)
            if self.angle != 0:
                gc.rotate_ctm(self.angle * 3.14159 / 180.0)
            gc.set_stroke_color(self.color_)
            gc.set_line_width(1.0)
            gc.move_to(-self.size, 0)
            gc.line_to(self.size, 0)
            gc.move_to(0, -self.size)
            gc.line_to(0, self.size)
            gc.stroke_path()


class Demo(DemoFrame):
    def _create_component(self):
        canvas = DropCanvas(bgcolor="lightsteelblue", draw_axes=True)
        canvas.overlays.append(
            EventTracer(canvas, color="green", size=8, angle=45.0)
        )

        viewport = Viewport(component=canvas, enable_zoom=True)
        viewport.view_position = [0, 0]
        viewport.tools.append(ViewportPanTool(viewport, drag_button="right"))
        viewport.overlays.append(EventTracer(viewport))

        scrolled = Scrolled(
            canvas,
            inside_padding_width=0,
            mousewheel_scroll=False,
            viewport_component=viewport,
            always_show_sb=True,
            continuous_drag_update=True,
        )
        return scrolled


if __name__ == "__main__":
    # Save demo so that it doesn't get garbage collected when run within
    # existing event loop (i.e. from ipython).
    demo = demo_main(Demo, title="Canvas example")
