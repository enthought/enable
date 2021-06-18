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
Example application for exploring Sierpinski's Triangle.
"""
import numpy as np

from traits.api import (
    HasTraits,
    Instance,
    Int,
    observe,
    Range,
    Property
)
from traitsui.api import Item, UItem, View
from enable.api import (
    Canvas, Component, ComponentEditor, Viewport, Scrolled
)
from enable.tools.api import ViewportPanTool

SQRT3 = np.sqrt(3)


class SierpinskiTriangle(Component):

    base_width = Int()

    iterations = Int()

    def _draw_mainlayer(self, gc, view_bounds=None, mode="default"):
        # draw the base triangle
        gc.translate_ctm(0., 0.)
        gc.set_fill_color((1.0, 1.0, 1.0, 1.0))
        gc.move_to(0, 0)
        gc.line_to(self.base_width, 0)
        gc.line_to(self.base_width/2, SQRT3*(self.base_width/2))
        gc.line_to(0, 0)
        gc.fill_path()

        gc.set_fill_color((0.0, 0.0, 0.0, 1.0))

        gc.begin_path()
        path = gc.get_empty_path()

        if self.iterations >= 1:
            self.add_triangle_to_path(
                path,
                (self.base_width/4, (self.base_width/4)*SQRT3),
                self.base_width/2
            )
            self.sierpinski(
                path,
                (self.base_width/4, (self.base_width/4)*SQRT3),
                1
            )

            gc.add_path(path)
            gc.fill_path()

    def sierpinski(self, path, point, n):
        """ Recursive method to determine size and location of triangles to
        add to input path object, based on the current iteration n.

        This method recursively adds to the path object which can be drawn upon
        completion of the full recursive call.
        """
        size = self.base_width/4 * (1/2)**(n - 1)
        if n == self.iterations:
            return

        # find top left corners of next 3 inverted triangles relative to
        # the top left corner of the current one (ie point)
        #
        # In the diagram below, X is point, and the ? represent the next
        # points to make recursive calls at
        #
        #       /\
        #     ?/__\
        #    X/_\/_\
        #    /\    /\
        #  ?/__\ ?/__\
        #  /_\/_\/_\/_\
        #

        rel_to_point_locs = (size/2)*np.array([
            [-1, -SQRT3],
            [1, SQRT3],
            [3, -SQRT3]
        ])
        # absolute location of those next points
        abs_points = point + rel_to_point_locs

        for point in abs_points:
            self.add_triangle_to_path(
                path,
                point,
                size
            )
            self.sierpinski(
                path, point, n+1
            )

    def add_triangle_to_path(self, path, point, size):
        """ Adds an inverted triangle to the input path, of side length size,
        using the input point as the upper left corner of the triangle.
        """
        x, y = point
        path.move_to(x, y)
        path.line_to(x + size, y)
        path.line_to(x + .5 * size, y - (SQRT3 / 2) * size)
        path.line_to(x, y)


class Viewer(HasTraits):

    iterations = Range(0, 'max_iters')

    triangle = Instance(Component)

    comp = Instance(Component)

    base_width = Int(500)

    max_iters = Property(observe="base_width, viewport.zoom")

    viewport = Instance(Viewport)

    def _get_bounds(self):
        return [self.base_width, self.base_width*(SQRT3/2)]

    def _get_max_iters(self):
        factor = 1
        if self.viewport:
            factor = self.viewport.zoom
        return int(np.log(4/(factor*self.base_width))/np.log(.5) + 1)

    def _comp_default(self):
        canvas = Canvas(draw_axes=True, bgcolor="gray")
        self.triangle = SierpinskiTriangle(
            position=[0.0, 0.0],
            bounds=[self.base_width, self.base_width*(SQRT3/2)],
            iterations=self.iterations,
            max_iters=self.max_iters,
            base_width=self.base_width,
            bgcolor="gray"
        )
        canvas.add(self.triangle)

        self.viewport = Viewport(
            component=canvas, enable_zoom=True, stay_inside=True
        )
        self.viewport.zoom_tool.min_zoom = 1.0
        self.viewport.tools.append(
            ViewportPanTool(self.viewport, drag_button="right")
        )

        scrolled = Scrolled(
            canvas,
            inside_padding_width=0,
            mousewheel_scroll=False,
            viewport_component=self.viewport,
            always_show_sb=True,
            continuous_drag_update=True,
        )
        return scrolled

    @observe("base_width")
    def _update_base_width(self, event):
        self.triangle.bounds = [event.new, event.new*(SQRT3/2)]
        self.triangle.base_width = event.new
        self.triangle.invalidate_and_redraw()

    @observe("iterations")
    def _redraw(self, event):
        self.triangle.iterations = event.new
        self.triangle.invalidate_and_redraw()

    traits_view = View(
        Item(name='iterations'),
        Item(name='base_width'),
        UItem(
            "comp",
            editor=ComponentEditor(size=(500, 500)),
            resizable=True
        ),
        resizable=True,
        buttons=["OK"],
        title="Sierpinski's triangle",
    )


if __name__ == '__main__':
    viewer = Viewer()
    viewer.configure_traits()
