""" A point-to-point drawn polygon. """

from __future__ import with_statement

from enable.api import cursor_style_trait, Line
from traits.api import Event, Int, Instance

from .drawing_tool import DrawingTool


class PointLine(DrawingTool):
    """ A point-to-point drawn line. """

    # Our contained "Line" instance; it stores the points and does the actual
    # drawing.
    line = Instance(Line, args=())

    # Override the draw_mode value we inherit from DrawingTool
    draw_mode = "overlay"

    # The pixel distance from a vertex that is considered 'on' the vertex.
    proximity_distance = Int(4)

    # The cursor shapes to use for various modes
    normal_cursor = cursor_style_trait('arrow')
    drawing_cursor = cursor_style_trait('pencil')
    delete_cursor = cursor_style_trait('bullseye')
    move_cursor = cursor_style_trait('sizing')

    # The index of the vertex being dragged, if any.
    _dragged = Int

    complete = Event

    def add_point(self, point):
        """ Add the point. """
        self.line.points.append(point)
        return

    def get_point(self, index):
        """ Get the point at the specified index. """
        return self.line.points[ index ]

    def set_point(self, index, point):
        """ Set the point at the specified index to point. """
        self.line.points[index] = point
        return

    def remove_point(self, index):
        """ Remove the point with the specified index. """
        del self.line.points[index]
        return

    #------------------------------------------------------------------------
    # DrawingTool interface
    #------------------------------------------------------------------------

    def reset(self):
        self.line.points = []
        self.event_state = "normal"
        return

    #------------------------------------------------------------------------
    # "complete" state
    #------------------------------------------------------------------------

    def complete_draw(self, gc):
        # Draw the completed line
        self.line.line_dash = None
        with gc:
            self.line._draw_mainlayer(gc)
        return

    def complete_left_down(self, event):
        """ Handle the left mouse button going down in the 'complete' state. """

        # Ignore the click if it contains modifiers we do not handle.
        if event.shift_down or event.alt_down:
            event.handled = False

        else:
            # If we are over a point, we will either move it or remove it.
            over = self._over_point(event, self.line.points)
            if over is not None:
                # Control down means remove it.
                if event.control_down:
                    self.remove_point(over)
                    self.updated = self

                # Otherwise, prepare to drag it.
                else:
                    self._dragged = over
                    event.window.set_pointer(self.move_cursor)
                    self.event_state = 'drag_point'
                    self.request_redraw()
        return

    def complete_mouse_move(self, event):
        """ Handle the mouse moving in the 'complete' state. """
        # If we are over a point, then we have to prepare to move it.
        over = self._over_point(event, self.line.points)
        if over is not None:
            if event.control_down:
                event.window.set_pointer(self.delete_cursor)
            else:
                event.window.set_pointer(self.move_cursor)
        else:
            event.handled = False
            event.window.set_pointer(self.normal_cursor)
        self.request_redraw()
        return

    #------------------------------------------------------------------------
    # "drag" state
    #------------------------------------------------------------------------

    def drag_point_draw(self, gc):
        """ Draw the polygon in the 'drag_point' state. """
        self.line._draw_mainlayer(gc)
        return

    def drag_point_left_up(self, event):
        """ Handle the left mouse coming up in the 'drag_point' state. """
        self.event_state = 'complete'
        self.updated = self
        return

    def drag_point_mouse_move(self, event):
        """ Handle the mouse moving in the 'drag_point' state. """
        # Only worry about the event if it's inside our bounds.
        dragged_point = self.get_point(self._dragged)
        # If the point has actually moved, update it.
        if dragged_point != (event.x, event.y):
            self.set_point(self._dragged, (event.x, event.y))
            self.request_redraw()
        return

    #------------------------------------------------------------------------
    # "incomplete" state
    #------------------------------------------------------------------------

    def incomplete_draw(self, gc):
        """ Draw the line in the 'incomplete' state. """
        with gc:
            gc.set_fill_color((0, 0, 0, 0))
            gc.rect(50, 50, 100, 100)
        self.line._draw_mainlayer(gc)
        return

    def incomplete_left_dclick(self, event):
        """ Handle a left double-click in the incomplete state. """
        # Remove the point that was placed by the first mouse down, since
        # another one will be placed on the down stroke of the double click.
        self.remove_point(-1)
        event.window.set_pointer(self.move_cursor)
        self.event_state = 'complete'
        self.complete = True
        self.request_redraw()
        return

    def incomplete_left_down(self, event):
        """ Handle the left mouse button coming up in incomplete state. """
        # Add the point.
        self.add_point((event.x, event.y))
        self.updated = self
        return

    def incomplete_mouse_move(self, event):
        """ Handle the mouse moving in incomplete state. """
        # If we move over the initial point, then we change the cursor.
        event.window.set_pointer(self.drawing_cursor)

        # If the point has actually changed, then we need to update our model.
        if self.get_point(-1) != (event.x, event.y):
            self.set_point(-1, (event.x, event.y))
        self.request_redraw()
        return

    #------------------------------------------------------------------------
    # "normal" state
    #------------------------------------------------------------------------

    def normal_left_down(self, event):
        """ Handle the left button up in the 'normal' state. """

        # Append the current point twice, because we need to have the starting
        # point and the current point be separate, since the current point
        # will be moved with the mouse from now on.
        self.add_point((event.x, event.y))
        self.add_point((event.x, event.y))
        self.event_state = 'incomplete'
        self.updated = self
        self.line_dash = (4.0, 2.0)
        return

    def normal_mouse_move(self, event):
        """ Handle the mouse moving in the 'normal' state. """
        event.window.set_pointer(self.drawing_cursor)
        return

    #------------------------------------------------------------------------
    # Private interface
    #------------------------------------------------------------------------

    def _updated_fired(self, event):
        # The self.updated trait is used by point_line and can be used by
        # others to indicate that the model has been updated.  For now, the
        # only action taken is to do a redraw.
        self.request_redraw()

    def _is_near_point(self, point, event):
        """ Determine if the pointer is near a specified point. """
        event_point = (event.x, event.y)

        return ((abs( point[0] - event_point[0] ) + \
                 abs( point[1] - event_point[1] )) <= self.proximity_distance)

    def _is_over_start(self, event):
        """ Test if the event is 'over' the starting vertex. """
        return (len(self.points) > 0 and
                self._is_near_point(self.points[0], event))

    def _over_point(self, event, points):
        """ Return the index of a point in points that event is 'over'.

        Returns None if there is no such point.
        """
        for i, point in enumerate(points):
            if self._is_near_point(point, event):
                result = i
                break
        else:
            result = None
        return result

# EOF
