# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
""" A line segment component. """

from numpy import array, resize

# Enthought library imports.
from kiva.api import FILL, FILL_STROKE, STROKE
from traits.api import Any, Event, Float, List, Trait, Bool

# Local imports.
from enable.api import border_size_trait, Component
from enable.colors import ColorTrait


class Line(Component):
    """A line segment component"""

    # Event fired when the points are no longer updating.
    # PZW: there seems to be a missing defn here; investigate.

    # An event to indicate that the point list has changed
    updated = Event

    # The color of the line.
    line_color = ColorTrait("black")

    # The dash pattern for the line.
    line_dash = Any

    # The width of the line.
    line_width = Trait(1, border_size_trait)

    # The points that make up this polygon.
    points = List  # List of Tuples

    # The color of each vertex.
    vertex_color = ColorTrait("black")

    # The size of each vertex.
    vertex_size = Float(3.0)

    # Whether to draw the path closed, with a line back to the first point
    close_path = Bool(True)

    # -------------------------------------------------------------------------
    # 'Line' interface
    # -------------------------------------------------------------------------

    def reset(self):
        "Reset the polygon to the initial state"

        self.points = []
        self.event_state = "normal"
        self.updated = self

    # -------------------------------------------------------------------------
    # 'Component' interface
    # -------------------------------------------------------------------------

    def _draw_mainlayer(self, gc, view_bounds=None, mode="default"):
        "Draw this line in the specified graphics context"
        if len(self.points) > 1:
            with gc:
                # Set the drawing parameters.
                gc.set_stroke_color(self.line_color_)
                gc.set_line_dash(self.line_dash)
                gc.set_line_width(self.line_width)

                # Draw the path as lines.
                gc.begin_path()
                offset_points = [(x, y) for x, y in self.points]
                offset_points = resize(
                    array(offset_points), (len(self.points), 2)
                )
                gc.lines(offset_points)
                if self.close_path:
                    gc.close_path()
                gc.draw_path(STROKE)

        if len(self.points) > 0:
            with gc:
                # Draw the vertices.
                self._draw_points(gc)

    # -------------------------------------------------------------------------
    # Private interface
    # -------------------------------------------------------------------------

    def _draw_points(self, gc):
        "Draw the points of the line"

        # Shortcut out if we would draw transparently.
        if self.vertex_color_[3] != 0:
            with gc:
                gc.set_fill_color(self.vertex_color_)
                gc.set_line_dash(None)

                offset_points = [(x, y) for x, y in self.points]
                offset_points = resize(
                    array(offset_points), (len(self.points), 2)
                )
                offset = self.vertex_size / 2.0
                if hasattr(gc, "draw_path_at_points"):
                    path = gc.get_empty_path()
                    path.rect(
                        -offset, -offset, self.vertex_size, self.vertex_size
                    )
                    gc.draw_path_at_points(offset_points, path, FILL_STROKE)
                else:
                    for x, y in offset_points:
                        gc.draw_rect(
                            (
                                x - offset,
                                y - offset,
                                self.vertex_size,
                                self.vertex_size,
                            ),
                            FILL,
                        )
