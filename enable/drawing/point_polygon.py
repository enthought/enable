# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
""" A point-to-point drawn polygon. """

# Enthought library imports.
from enable.primitives.api import Polygon
from traits.api import Int, Instance

from .drawing_tool import DrawingTool


class PointPolygon(DrawingTool):
    """ A point-to-point drawn polygon. """

    # The actual polygon primitive we are editing.
    polygon = Instance(Polygon, args=())

    # The pixel distance from a vertex that is considered 'on' the vertex.
    proximity_distance = Int(4)

    # Override the default value of this inherited trait
    draw_mode = "overlay"

    # The index of the vertex being dragged, if any.
    _dragged = Int

    def reset(self):
        self.polygon.model.reset()
        self.event_state = "normal"

    # ------------------------------------------------------------------------
    # "complete" state
    # ------------------------------------------------------------------------

    def complete_draw(self, gc):
        """ Draw a closed polygon. """
        self.polygon.border_dash = None
        self.polygon._draw_closed(gc)

    def complete_left_down(self, event):
        """ Handle the left mouse button coming up in the 'complete' state. """
        # Ignore the click if it contains modifiers we do not handle.
        polygon = self.polygon
        if event.shift_down or event.alt_down:
            event.handled = False
        else:
            # If we are over a point, we will either move it or remove it.
            over = self._over_point(event, polygon.model.points)
            if over is not None:
                # Control down means remove it.
                if event.control_down:
                    del polygon.model.points[over]

                # Otherwise, prepare to drag it.
                else:
                    self._dragged = over
                    event.window.set_pointer("right arrow")
                    self.event_state = "drag_point"

                self.request_redraw()

    def complete_mouse_move(self, event):
        """ Handle the mouse moving in the 'complete' state. """

        # If we are over a point, then we have to prepare to move it.
        over = self._over_point(event, self.polygon.model.points)
        if over is not None:
            if event.control_down:
                event.window.set_pointer("bullseye")
            else:
                event.window.set_pointer("right arrow")
        else:
            event.handled = False
            event.window.set_pointer("arrow")
        self.request_redraw()

    # ------------------------------------------------------------------------
    # "drag_point" state
    # ------------------------------------------------------------------------

    def drag_point_draw(self, gc):
        """ Draw the polygon in the 'drag_point' state. """
        self.complete_draw(gc)

    def drag_point_left_up(self, event):
        """ Handle the left mouse coming up in the 'drag_point' state. """
        self.event_state = "complete"
        self.request_redraw()

    def drag_point_mouse_move(self, event):
        """ Handle the mouse moving in the 'drag_point' state. """
        # Only worry about the event if it's inside our bounds.
        polygon = self.polygon
        dragged_point = polygon.model.points[self._dragged]

        # If the point has actually moved, update it.
        if dragged_point != (event.x, event.y):
            polygon.model.points[self._dragged] = (
                event.x + self.x,
                event.y - self.y,
            )
            self.request_redraw()

    # ------------------------------------------------------------------------
    # "incomplete" state
    # ------------------------------------------------------------------------

    def incomplete_draw(self, gc):
        """ Draw the polygon in the 'incomplete' state. """
        self.polygon.border_dash = (4.0, 2.0)
        self.polygon._draw_open(gc)

    def incomplete_left_dclick(self, event):
        """ Handle a left double-click in the incomplete state. """

        # Remove the point that was placed by the first mouse up, since
        # another one will be placed on the up stroke of the double click.
        del self.polygon.model.points[-1]
        event.window.set_pointer("right arrow")
        self.event_state = "complete"
        self.complete = True

        self.request_redraw()

    def incomplete_left_up(self, event):
        """ Handle the left mouse button coming up in incomplete state. """

        # If the click was over the start vertex, we are done.
        if self._is_over_start(event):
            del self.polygon.model.points[-1]
            self.event_state = "complete"
            event.window.set_pointer("right arrow")
            self.complete = True

        # Otherwise, add the point and move on.
        else:
            self.polygon.model.points.append(
                (event.x + self.x, event.y - self.y)
            )

        self.request_redraw()

    def incomplete_mouse_move(self, event):
        """ Handle the mouse moving in incomplete state. """
        # If we move over the initial point, then we change the cursor.
        if self._is_over_start(event):
            event.window.set_pointer("bullseye")
        else:
            event.window.set_pointer("pencil")

        # If the point has actually changed, then we need to update our model.
        if self.polygon.model.points != (event.x + self.x, event.y - self.y):
            self.polygon.model.points[-1] = (
                event.x + self.x,
                event.y - self.y,
            )

        self.request_redraw()

    # ------------------------------------------------------------------------
    # "normal" state
    # ------------------------------------------------------------------------

    def normal_left_up(self, event):
        """ Handle the left button up in the 'normal' state. """

        # Append the current point twice, because we need to have the starting
        # point and the current point be separate, since the current point
        # will be moved with the mouse from now on.
        pt = (event.x + self.x, event.y - self.y)
        self.polygon.model.points.append(pt)
        self.polygon.model.points.append(pt)
        self.event_state = "incomplete"

    def normal_mouse_move(self, event):
        """ Handle the mouse moving in the 'normal' state. """
        event.window.set_pointer("pencil")

    # ------------------------------------------------------------------------
    # private methods
    # ------------------------------------------------------------------------

    def _is_near_point(self, point, event):
        """ Determine if the pointer is near a specified point. """
        event_point = (event.x + self.x, event.y - self.y)
        return (
            abs(point[0] - event_point[0]) + abs(point[1] - event_point[1])
        ) <= self.proximity_distance

    def _is_over_start(self, event):
        """ Test if the event is 'over' the starting vertex. """
        return len(self.polygon.model.points) > 0 and self._is_near_point(
            self.polygon.model.points[0], event
        )

    def _over_point(self, event, points):
        """ Return the index of a point in points that event is 'over'.

        Returns none if there is no such point.
        """
        for i, point in enumerate(points):
            if self._is_near_point(point, event):
                result = i
                break
        else:
            result = None
        return result
