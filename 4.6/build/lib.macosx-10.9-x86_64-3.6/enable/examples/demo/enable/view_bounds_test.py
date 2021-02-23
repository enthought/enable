# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
""" Demonstrates how clipping of objects occurs with the view_bounds parameter
to draw().
"""

from enable.api import Container, Component, Scrolled
from enable.base import empty_rectangle, intersect_bounds
from enable.example_support import DemoFrame, demo_main


class Box(Component):
    def _draw_mainlayer(self, gc, view_bounds=None, mode="default"):
        with gc:
            gc.set_fill_color((1.0, 0.0, 0.0, 1.0))
            dx, dy = self.bounds
            x, y = self.position
            gc.rect(x, y, dx, dy)
            gc.fill_path()


class VerboseContainer(Container):
    """
    By default a container just doesn't draw contained components if they are
    outside its view_bounds.  This subclass modifies the behavior to report
    what components are not being drawn and what their bounds are.
    """

    fit_window = False

    def _draw_container_mainlayer(self, gc, view_bounds, mode="default"):
        with gc:
            gc.set_fill_color((1.0, 1.0, 1.0, 1.0))
            gc.set_stroke_color((1.0, 1.0, 1.0, 1.0))
            gc.rect(self.x, self.y, self.width, self.height)
            gc.fill_path()

        if view_bounds:
            v = view_bounds
            new_bounds = (v[0] - self.x, v[1] - self.y, v[2], v[3])
        else:
            new_bounds = None

        with gc:
            gc.translate_ctm(*self.position)
            gc.set_stroke_color((0.0, 0.0, 0.0, 1.0))
            for component in self._components:
                # See if the component is visible:
                tmp = intersect_bounds(
                    component.position + component.bounds, new_bounds
                )
                if tmp == empty_rectangle:
                    print(
                        "skipping component:",
                        component.__class__.__name__,
                        end=" ",
                    )
                    print("\tbounds:", component.position, component.bounds)
                    continue

                with gc:
                    component.draw(gc, new_bounds, mode)


class Demo(DemoFrame):
    def _create_component(self):
        container = VerboseContainer(auto_size=False, bounds=[800, 800])
        a = Box(bounds=[50.0, 50.0], position=[50.0, 50.0])
        b = Box(bounds=[50.0, 50.0], position=[200.0, 50.0])
        c = Box(bounds=[50.0, 50.0], position=[50.0, 200.0])
        d = Box(bounds=[50.0, 50.0], position=[200.0, 200.0])
        container.add(a)
        container.add(b)
        container.add(c)
        container.add(d)
        scr = Scrolled(
            container, bounds=[300, 300], position=[50, 50], fit_window=False
        )
        return scr


if __name__ == "__main__":
    title = "Use the scroll bars to show and hide components"
    # Save demo so that it doesn't get garbage collected when run within
    # existing event loop (i.e. from ipython).
    demo = demo_main(Demo, title=title)
