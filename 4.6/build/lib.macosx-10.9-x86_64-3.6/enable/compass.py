# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
from numpy import array, pi

# Enthought library imports
from traits.api import Bool, Enum, Float, Int

# Local, relative imports
from .component import Component
from .colors import ColorTrait


class Compass(Component):
    """ A compass with triangles at the 4 cardinal directions.  The center
    of the compass of triangles is the center of the widget.
    """

    # Which triangle was clicked
    clicked = Enum(None, "n", "e", "s", "w", "c")

    # Whether or not to allow clicks on the center
    enable_center = Bool(False)

    # ------------------------------------------------------------------------
    # Shape and layout
    # ------------------------------------------------------------------------

    # The length of the triangle from tip to base
    triangle_length = Int(11)

    # The width of the base of the triangle
    triangle_width = Int(8)

    # Overall scaling factor for the triangle.  Note that this also scales
    # the outline width.
    scale = Float(1.0)

    # The distance from the center of the widget to the center of each triangle
    # (halfway along its length, not the orthocenter).
    spacing = Int(12)

    # ------------------------------------------------------------------------
    # Appearance Traits
    # ------------------------------------------------------------------------

    # The line color of the triangles
    color = ColorTrait("black")

    # The line width of the triangles
    line_width = Int(2)

    # The triangle fill color when the mouse has not been clicked
    fill_color = ColorTrait("none")

    # The fill color of the triangle that the user has clicked on
    clicked_color = ColorTrait("lightgray")

    # Override the inherited **event_state** attribute
    event_state = Enum("normal", "clicked")

    # ------------------------------------------------------------------------
    # Stub methods for subclasses
    # ------------------------------------------------------------------------

    def mouse_down(self, arrow):
        """ Called when the mouse is first pressed inside one of the
        triangles.  This gets called after self.clicked is set.

        Parameters
        ==========
        arrow: "n", "e", "s", "w"
            indicates which arrow was pressed
        """
        pass

    def mouse_up(self):
        """ Called when the mouse is released.  This gets called after
        self.clicked is unset.
        """
        pass

    # ------------------------------------------------------------------------
    # Event handling methods
    # ------------------------------------------------------------------------

    def normal_left_down(self, event):
        # Determine which arrow was clicked; use a rectangular approximation.
        x = event.x - (self.x + self.width / 2)
        y = event.y - (self.y + self.height / 2)
        half_length = self.triangle_length / 2 * self.scale
        half_width = self.triangle_width / 2 * self.scale
        offset = self.spacing * self.scale

        # Create dict mapping direction to (x, y, x2, y2)
        near = offset - half_length
        far = offset + half_length
        rects = {
            "n": array((-half_width, near, half_width, far)),
            "e": array((near, -half_width, far, half_width)),
            "s": array((-half_width, -far, half_width, -near)),
            "w": array((-far, -half_width, -near, half_width)),
        }
        if self.enable_center:
            rects["c"] = array((-near, -near, near, near))
        for direction, rect in rects.items():
            if (rect[0] <= x <= rect[2]) and (rect[1] <= y <= rect[3]):
                self.event_state = "clicked"
                self.clicked = direction
                self.mouse_down(direction)
                self.request_redraw()
                break
        event.handled = True

    def normal_left_dclick(self, event):
        return self.normal_left_down(event)

    def clicked_left_up(self, event):
        self.event_state = "normal"
        self.clicked = None
        event.handled = True
        self.mouse_up()
        self.request_redraw()

    def clicked_mouse_leave(self, event):
        self.clicked_left_up(event)

    # ------------------------------------------------------------------------
    # Rendering methods
    # ------------------------------------------------------------------------

    def get_preferred_size(self):
        # Since we can compute our preferred size from the size of the
        # arrows and the spacing, we can return a sensible preferred
        # size, so override the default implementation in Component.

        if self.fixed_preferred_size is not None:
            return self.fixed_preferred_size
        else:
            extent = self.scale * 2 * (self.spacing + self.triangle_length / 2)
            return [extent + self.hpadding, extent + self.vpadding]

    def _draw_mainlayer(self, gc, view_bounds=None, mode="normal"):
        with gc:
            gc.set_stroke_color(self.color_)
            gc.set_line_width(self.line_width)
            gc.translate_ctm(self.x + self.width / 2, self.y + self.height / 2)
            s = self.spacing
            points_and_angles = [
                ("n", (0, s), 0),
                ("e", (s, 0), -pi / 2),
                ("s", (0, -s), pi),
                ("w", (-s, 0), pi / 2),
            ]

            gc.scale_ctm(self.scale, self.scale)
            for dir, (dx, dy), angle in points_and_angles:
                if self.event_state == "clicked" and self.clicked == dir:
                    gc.set_fill_color(self.clicked_color_)
                else:
                    gc.set_fill_color(self.fill_color_)
                gc.translate_ctm(dx, dy)
                gc.rotate_ctm(angle)
                half_height = self.triangle_length / 2
                half_width = self.triangle_width / 2
                gc.begin_path()
                gc.lines(
                    [
                        (-half_width, -half_height),
                        (0, half_height),
                        (half_width, -half_height),
                        (-half_width, -half_height),
                        (0, half_height),
                    ]
                )
                gc.draw_path()
                gc.rotate_ctm(-angle)
                gc.translate_ctm(-dx, -dy)

            if self.event_state == "clicked" and self.clicked == "c":
                # Fill in the center
                gc.set_fill_color(self.clicked_color_)
                half_width = self.triangle_width / 2
                gc.begin_path()
                gc.lines(
                    [
                        (-half_width, -half_width),
                        (half_width, -half_height),
                        (half_width, half_width),
                        (-half_width, half_width),
                        (-half_width, -half_width),
                    ]
                )
                gc.draw_path()
