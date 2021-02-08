# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
""" A drag drawn polygon. """

from enable.primitives.api import Polygon
from enable.api import Pointer
from pyface.action.api import MenuManager
from traits.api import Delegate, Instance

from .drawing_tool import DrawingTool


class DragPolygon(DrawingTool):
    """ A drag drawn polygon. """

    poly = Instance(Polygon, args=())

    draw_mode = "overlay"

    # Visible style. ####

    # Override the vertex color so as to not draw it.
    vertex_color = Delegate("poly", modify=True)

    # Override the vertex size so as to not draw it.
    vertex_size = Delegate("poly", modify=True)

    background_color = Delegate("poly", modify=True)

    # Pointers. ####

    # Pointer for the complete state.
    complete_pointer = Pointer("cross")

    # Pointer for the drawing state.
    drawing_pointer = Pointer("cross")

    # Pointer for the normal state.
    normal_pointer = Pointer("cross")

    # Miscellaneous. ####

    # The context menu for the polygon.
    menu = Instance(MenuManager)

    def reset(self):
        self.vertex_color = (0, 0, 0, 0)
        self.vertex_size = 0
        self.poly.model.points = []
        self.event_state = "normal"

    ###########################################################################
    # 'Component' interface.
    ###########################################################################

    # 'complete' state #####################################################

    def complete_draw(self, gc):
        """ Draw the completed polygon. """
        with gc:
            self.poly.border_dash = None
            self.poly._draw_closed(gc)

    def complete_left_down(self, event):
        """ Draw a new polygon. """
        self.reset()
        self.normal_left_down(event)

    def complete_right_down(self, event):
        """ Do the context menu if available. """
        if self.menu is not None:
            if self._is_in((event.x + self.x, event.y - self.y)):
                menu = self.menu.create_menu(event.window.control)
                # FIXME : The call to _flip_y is necessary but inappropriate.
                menu.show(event.x, event.window._flip_y(event.y))

    # 'drawing' state ######################################################

    def drawing_draw(self, gc):
        """ Draw the polygon while in 'drawing' state. """

        with gc:
            self.poly.border_dash = (4.0, 2.0)
            self.poly._draw_open(gc)

    def drawing_left_up(self, event):
        """ Handle the left mouse button coming up in 'drawing' state. """

        self.event_state = "complete"
        self.pointer = self.complete_pointer

        self.request_redraw()

        self.complete = True

    def drawing_mouse_move(self, event):
        """ Handle the mouse moving in 'drawing' state. """

        last_point = self.poly.model.points[-1]

        # If we have moved, we need to add a point.
        if last_point != (event.x + self.x, event.y - self.y):
            self.poly.model.points.append((event.x + self.x, event.y - self.y))
            self.request_redraw()

    # 'normal' state #######################################################

    def normal_left_down(self, event):
        """ Handle the left button down in the 'normal' state. """

        self.poly.model.points.append((event.x + self.x, event.y - self.y))

        self.event_state = "drawing"
        self.pointer = self.drawing_pointer

        self.request_redraw()

    def normal_mouse_move(self, event):
        """ Handle the mouse moving in the 'normal' state. """

        self.pointer = self.normal_pointer
