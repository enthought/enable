# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!

# Enthought library imports
from enable.api import Line, Pointer
from traits.api import Event, Instance

from .drawing_tool import DrawingTool


class DragSegment(DrawingTool):
    """A dragged line segment"""

    # Override the vertex color so as to not draw it.
    vertex_color = (0.0, 0.0, 0.0, 0.0)

    # Because this class subclasses DrawingTool and not Line, it contains
    # an instance of the Line primitive.
    line = Instance(Line, args=())

    # Event fired when the line is complete
    complete = Event

    # Pointer for the complete state.
    complete_pointer = Pointer("arrow")

    # Pointer for the drawing state.
    drawing_pointer = Pointer("cross")

    # Pointer for the normal state.
    normal_pointer = Pointer("cross")

    # ------------------------------------------------------------------------
    # DrawingTool interface
    # ------------------------------------------------------------------------

    def reset(self):
        self.line.vertex_color = self.vertex_color
        self.line.points = []
        self.event_state = "normal"

    # ------------------------------------------------------------------------
    # "complete" state
    # ------------------------------------------------------------------------

    def complete_draw(self, gc):
        """ Draw the completed line. """
        self.line.line_dash = None
        self.line._draw_mainlayer(gc)
        self.request_redraw()

    # ------------------------------------------------------------------------
    # "drawing" state
    # ------------------------------------------------------------------------

    def drawing_draw(self, gc):
        self.line.line_dash = (4.0, 2.0)
        self.line._draw_mainlayer(gc)

    def drawing_mouse_move(self, event):
        """ Handle the mouse moving in drawing state. """
        # Change the last point to the current event point
        self.line.points[-1] = (event.x, event.y)
        self.updated = self
        self.request_redraw()

    def drawing_left_up(self, event):
        """ Handle the left mouse button coming up in the 'drawing' state. """
        self.event_state = "complete"
        event.window.set_pointer(self.complete_pointer)
        self.request_redraw()
        self.complete = True

    # ------------------------------------------------------------------------
    # "normal" state
    # ------------------------------------------------------------------------

    def normal_left_down(self, event):
        """ Handle the left button down in the 'normal' state. """
        # Set points the current segment, which is just the
        # current point twice.
        current_point = (event.x, event.y)
        self.line.points = [current_point, current_point]
        self.updated = self

        # Go into the drawing state
        self.event_state = "drawing"
        event.window.set_pointer(self.drawing_pointer)

        self.request_redraw()

    def normal_mouse_move(self, event):
        event.window.set_pointer(self.normal_pointer)
