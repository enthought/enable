#-----------------------------------------------------------------------------
#
#  Copyright (c) 2005, Enthought, Inc.
#  All rights reserved.
#
#  Author: Scott Swarts <swarts@enthought.com>
#
#-----------------------------------------------------------------------------

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
    complete_pointer = Pointer('arrow')

    # Pointer for the drawing state.
    drawing_pointer = Pointer('cross')

    # Pointer for the normal state.
    normal_pointer = Pointer('cross')

    #------------------------------------------------------------------------
    # DrawingTool interface
    #------------------------------------------------------------------------

    def reset(self):
        self.line.vertex_color = self.vertex_color
        self.line.points = []
        self.event_state = "normal"
        return

    #------------------------------------------------------------------------
    # "complete" state
    #------------------------------------------------------------------------

    def complete_draw(self, gc):
        """ Draw the completed line. """
        self.line.line_dash = None
        self.line._draw_mainlayer(gc)
        self.request_redraw()
        return

    #------------------------------------------------------------------------
    # "drawing" state
    #------------------------------------------------------------------------

    def drawing_draw(self, gc):
        self.line.line_dash = (4.0, 2.0)
        self.line._draw_mainlayer(gc)
        return

    def drawing_mouse_move(self, event):
        """ Handle the mouse moving in drawing state. """
        # Change the last point to the current event point
        self.line.points[-1] = (event.x, event.y)
        self.updated = self
        self.request_redraw()
        return

    def drawing_left_up(self, event):
        """ Handle the left mouse button coming up in the 'drawing' state. """
        self.event_state = 'complete'
        event.window.set_pointer(self.complete_pointer)
        self.request_redraw()
        self.complete = True
        return

    #------------------------------------------------------------------------
    # "normal" state
    #------------------------------------------------------------------------

    def normal_left_down(self, event):
        """ Handle the left button down in the 'normal' state. """
        # Set points the current segment, which is just the
        # current point twice.
        current_point = (event.x, event.y)
        self.line.points = [current_point, current_point]
        self.updated = self

        # Go into the drawing state
        self.event_state = 'drawing'
        event.window.set_pointer(self.drawing_pointer)

        self.request_redraw()
        return

    def normal_mouse_move(self, event):
        event.window.set_pointer(self.normal_pointer)
        return

# EOF
