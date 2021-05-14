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
    bounds_trait, Component, ComponentEditor, Inverted_TriangleMarker
)


marker = Inverted_TriangleMarker()
SQRT3 = np.sqrt(3)


class SierpinskiTriangle(Component):

    base_width = Int()

    iterations = Int()

    def _draw_mainlayer(self, gc, view_bounds=None, mode="default"):
        
        # draw the base triangle
        gc.translate_ctm(0., 0.)
        gc.set_fill_color((1.0, 1.0, 1.0, 1.0))
        gc.move_to(0,0)
        gc.line_to(self.base_width,0)
        gc.line_to(self.base_width/2, SQRT3*(self.base_width/2))
        gc.line_to(0,0)
        gc.fill_path()

        gc.set_fill_color((0.0, 0.0, 0.0, 1.0))

        if self.iterations >= 1:
            with gc:
                # draw first inverted triangle
                path = gc.get_empty_path()
                marker.add_to_path(path, self.base_width/4)
                gc.draw_path_at_points(
                    [[self.base_width/2, (self.base_width/4)*(SQRT3 - 1)]],
                    path,
                    marker.draw_mode
                )
                point_contexts = self.sierpinski(
                    (self.base_width, (self.base_width/2)*(SQRT3 - 1)),
                    {},
                    2
                )

            for size in point_contexts:
                path = gc.get_empty_path()
                marker.add_to_path(path, size)
                gc.draw_path_at_points(
                    point_contexts[size], path, marker.draw_mode
                )

    
    def sierpinski(self, center_loc, point_contexts, n):
        """ Recursive method to find locations / sizes of triangles to be
        drawn given the current number of iterations. The method modifies and
        returns the point_contexts input.
        """
        size = self.base_width/4 * (1/2)**(n - 1)
        if n == self.iterations + 1:
            return point_contexts
        else:
            # find centers of next 3 inverted triangles relative to the center
            # of the current one.
            rel_to_center_locs = 2*size*np.array([
                [-1, -(SQRT3 - 1)/2],
                [1, -(SQRT3 - 1)/2],
                [0, 1 + (SQRT3 - 1)/2]
            ])

            # absolute location of those centers
            abs_points = .5 * (center_loc + 2*rel_to_center_locs)

            if size in point_contexts:
                point_contexts[size] = np.append(
                    point_contexts[size], abs_points, 0
                )
            else:
                point_contexts[size] = abs_points

            for center_loc in abs_points:
                point_contexts = self.sierpinski(
                    2*center_loc, point_contexts, n+1
                )
            return point_contexts


class Viewer(HasTraits):

    iterations = Range(0, 'max_iters')

    triangle = Instance(Component)

    base_width = Int(500)

    max_iters = Property(observe="base_width")

    bgcolor = "black"

    def _get_max_iters(self):
        return int(np.log(4/self.base_width)/np.log(.5) + 1)

    def _triangle_default(self):
        tri = SierpinskiTriangle(
            position=[0.0, 0.0],
            bounds=[self.base_width, self.base_width*(SQRT3/2)],
            iterations=self.iterations,
            max_iters=self.max_iters,
            base_width=self.base_width,
            bgcolor=self.bgcolor,
        )
        return tri

    @observe("base_width")
    def _update_base_width(self, event):
        self.triangle.bounds = [self.base_width, self.base_width*(SQRT3/2)]
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
            "triangle",
            # fixme: make size automatically what  we want...
            editor=ComponentEditor(bgcolor="black", size=(500, 500)),
            resizable=True
        ),
        resizable=True,
        buttons=["OK"],
        title="Sierpinski's triangle",
    )


if __name__ == '__main__':
    viewer = Viewer()
    viewer.configure_traits()
