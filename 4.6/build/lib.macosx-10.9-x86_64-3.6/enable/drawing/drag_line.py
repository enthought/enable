# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
""" A drag drawn line. """

from enable.api import Line
from traits.api import Instance

from .drawing_tool import DrawingTool


class DragLine(DrawingTool):
    """
    A drag drawn line.  This is not a straight line, but can be a free-form,
    curved path.
    """

    # Override the vertex color so as to not draw it.
    vertex_color = (0.0, 0.0, 0.0, 0.0)

    # Because this class subclasses DrawingTool and not Line, it contains
    # an instance of the Line primitive.
    line = Instance(Line, args=())

    # Override the default value of this inherited trait
    draw_mode = "overlay"

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

    # ------------------------------------------------------------------------
    # "drawing" state
    # ------------------------------------------------------------------------

    def drawing_draw(self, gc):
        self.line.line_dash = (4.0, 2.0)
        self.line._draw_mainlayer(gc)

    def drawing_left_up(self, event):
        """ Handle the left mouse button coming up in the 'drawing' state. """
        self.event_state = "complete"
        event.window.set_pointer("arrow")
        self.request_redraw()
        self.complete = True
        event.handled = True

    def drawing_mouse_move(self, event):
        """ Handle the mouse moving in 'drawing' state. """
        last_point = self.line.points[-1]
        # If we have moved, we need to add a point.
        if last_point != (event.x + self.x, event.y - self.y):
            self.line.points.append((event.x + self.x, event.y - self.y))
            self.request_redraw()

    # ------------------------------------------------------------------------
    # "normal" state
    # ------------------------------------------------------------------------

    def normal_left_down(self, event):
        """ Handle the left button down in the 'normal' state. """

        self.line.points.append((event.x + self.x, event.y - self.y))
        self.event_state = "drawing"
        event.window.set_pointer("pencil")
        event.handled = True
        self.request_redraw()
