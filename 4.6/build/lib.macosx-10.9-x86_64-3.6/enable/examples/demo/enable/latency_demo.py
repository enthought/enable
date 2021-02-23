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
Test to see what level of click latency is noticeable.
"""

import time

from traits.api import Float

from enable.api import Component, Container, ColorTrait, black_color_trait
from enable.example_support import DemoFrame, demo_main
from kiva.api import SWISS, Font

font = Font(family=SWISS)


class Box(Component):
    color = ColorTrait("red")

    delay = Float(0.50)

    def _draw_mainlayer(self, gc, view=None, mode="default"):
        if self.event_state == "clicked":
            print("waiting %0.4f seconds... " % self.delay, end=" ")
            time.sleep(self.delay)
            print("done.")

            with gc:
                gc.set_fill_color(self.color_)
                gc.rect(*(self.position + self.bounds))
                gc.fill_path()

        else:
            with gc:
                gc.set_stroke_color(self.color_)
                gc.set_fill_color(self.color_)
                gc.set_line_width(1.0)
                gc.rect(*(self.position + self.bounds))
                gc.stroke_path()

                gc.set_font(font)
                x, y = self.position
                dx, dy = self.bounds
                tx, ty, tdx, tdy = gc.get_text_extent(str(self.delay))
                gc.set_text_position(
                    x + dx / 2 - tdx / 2, y + dy / 2 - tdy / 2
                )
                gc.show_text(str(self.delay))

    def normal_left_down(self, event):
        self.event_state = "clicked"
        event.handled = True
        self.request_redraw()

    def clicked_left_up(self, event):
        self.event_state = "normal"
        event.handled = True
        self.request_redraw()


class MyContainer(Container):
    text_color = black_color_trait

    def _draw_container_mainlayer(self, gc, view_bounds=None, mode="default"):
        s = "Hold down the mouse button on the boxes."
        with gc:
            gc.set_font(font)
            gc.set_fill_color(self.text_color_)
            tx, ty, tdx, tdy = gc.get_text_extent(s)
            x, y = self.position
            dx, dy = self.bounds
            gc.set_text_position(x + dx / 2 - tdx / 2, y + dy - tdy - 20)
            gc.show_text(s)


class Demo(DemoFrame):
    def _create_component(self):
        times_and_bounds = {
            0.5: (60, 200, 100, 100),
            0.33: (240, 200, 100, 100),
            0.25: (60, 50, 100, 100),
            0.10: (240, 50, 100, 100),
        }

        container = MyContainer(auto_size=False)
        for delay, bounds in list(times_and_bounds.items()):
            box = Box()
            container.add(box)
            box.position = list(bounds[:2])
            box.bounds = list(bounds[2:])
            box.delay = delay
        return container


if __name__ == "__main__":

    # Save demo so that it doesn't get garbage collected when run within
    # existing event loop (i.e. from ipython).
    demo = demo_main(Demo, size=(400, 400), title="Latency Test - Click a box")
