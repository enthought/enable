"""A filled polygon component"""

# Major package imports.
from numpy import array

# Enthought library imports.
from kiva.api import EOF_FILL_STROKE, FILL, FILL_STROKE, points_in_polygon
from traits.api import (
    Any, Event, Float, HasTraits, Instance, List, Property, Trait, Tuple
)
from traitsui.api import Group, View

# Local imports.
from enable.api import border_size_trait, Component
from enable.colors import ColorTrait


class PolygonModel(HasTraits):
    """ The data model for a Polygon. """

    # The points that make up the vertices of this polygon.
    points = List(Tuple)

    def reset(self):
        self.points = []


class Polygon(Component):
    """ A filled polygon component. """

    # -------------------------------------------------------------------------
    # Trait definitions.
    # -------------------------------------------------------------------------

    # The background color of this polygon.
    background_color = ColorTrait("white")

    # The color of the border of this polygon.
    border_color = ColorTrait("black")

    # The dash pattern to use for this polygon.
    border_dash = Any

    # The thickness of the border of this polygon.
    border_size = Trait(1, border_size_trait)

    # Event fired when the polygon is "complete".
    complete = Event

    # The rule to use to determine the inside of the polygon.
    inside_rule = Trait(
        "winding", {"winding": FILL_STROKE, "oddeven": EOF_FILL_STROKE}
    )

    # The points that make up this polygon.
    model = Instance(PolygonModel, ())

    # Convenience property to access the model's points.
    points = Property

    # The color of each vertex.
    vertex_color = ColorTrait("black")

    # The size of each vertex.
    vertex_size = Float(3.0)

    traits_view = View(
        Group("<component>", id="component"),
        Group("<links>", id="links"),
        Group(
            "background_color",
            "_",
            "border_color",
            "_",
            "border_size",
            id="Box",
            style="custom",
        ),
    )

    colorchip_map = {"color": "color", "alt_color": "border_color"}

    # -------------------------------------------------------------------------
    # Traits property accessors
    # -------------------------------------------------------------------------

    def _get_points(self):
        return self.model.points

    # -------------------------------------------------------------------------
    # 'Polygon' interface
    # -------------------------------------------------------------------------

    def reset(self):
        "Reset the polygon to the initial state"
        self.model.reset()
        self.event_state = "normal"

    # -------------------------------------------------------------------------
    # 'Component' interface
    # -------------------------------------------------------------------------

    def _draw_mainlayer(self, gc, view_bounds=None, mode="normal"):
        "Draw the component in the specified graphics context"
        self._draw_closed(gc)

    # -------------------------------------------------------------------------
    # Protected interface
    # -------------------------------------------------------------------------

    def _is_in(self, point):
        """ Test if the point (an x, y tuple) is within this polygonal region.

        To perform the test, we use the winding number inclusion algorithm,
        referenced in the comp.graphics.algorithms FAQ
        (http://www.faqs.org/faqs/graphics/algorithms-faq/) and described in
        detail here:

        http://softsurfer.com/Archive/algorithm_0103/algorithm_0103.htm
        """
        point_array = array((point,))
        vertices = array(self.model.points)
        winding = self.inside_rule == "winding"
        result = points_in_polygon(point_array, vertices, winding)
        return result[0]

    # -------------------------------------------------------------------------
    # Private interface
    # -------------------------------------------------------------------------

    def _draw_closed(self, gc):
        "Draw this polygon as a closed polygon"

        if len(self.model.points) > 2:
            # Set the drawing parameters.
            gc.set_fill_color(self.background_color_)
            gc.set_stroke_color(self.border_color_)
            gc.set_line_width(self.border_size)
            gc.set_line_dash(self.border_dash)

            # Draw the path.
            gc.begin_path()
            gc.move_to(
                self.model.points[0][0] - self.x,
                self.model.points[0][1] + self.y,
            )
            offset_points = [
                (x - self.x, y + self.y) for x, y in self.model.points
            ]
            gc.lines(offset_points)

            gc.close_path()
            gc.draw_path(self.inside_rule_)

            # Draw the vertices.
            self._draw_vertices(gc)

    def _draw_open(self, gc):
        "Draw this polygon as an open polygon"

        if len(self.model.points) > 2:
            # Set the drawing parameters.
            gc.set_fill_color(self.background_color_)
            gc.set_stroke_color(self.border_color_)
            gc.set_line_width(self.border_size)
            gc.set_line_dash(self.border_dash)

            # Draw the path.
            gc.begin_path()
            gc.move_to(
                self.model.points[0][0] - self.x,
                self.model.points[0][1] + self.y,
            )
            offset_points = [
                (x - self.x, y + self.y) for x, y in self.model.points
            ]
            gc.lines(offset_points)
            gc.draw_path(self.inside_rule_)

            # Draw the vertices.
            self._draw_vertices(gc)

    def _draw_vertices(self, gc):
        "Draw the vertices of the polygon."

        gc.set_fill_color(self.vertex_color_)
        gc.set_line_dash(None)

        offset = self.vertex_size / 2.0
        offset_points = [
            (x + self.x, y + self.y) for x, y in self.model.points
        ]

        if hasattr(gc, "draw_path_at_points"):
            path = gc.get_empty_path()
            path.rect(-offset, -offset, self.vertex_size, self.vertex_size)
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
