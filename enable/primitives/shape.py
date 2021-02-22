# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
""" The base class for moveable shapes. """

# Standard library imports.
import math

# Enthought library imports.
from enable.colors import ColorTrait
from enable.component import Component
from enable.enable_traits import Pointer
from kiva.api import MODERN, Font
from traits.api import Float, Property, Str, Tuple


class Shape(Component):
    """ The base class for moveable shapes. """

    # 'Component' interface ################################################

    # The background color of this component.
    bgcolor = "transparent"

    # 'Shape' interface ####################################################

    # The coordinates of the center of the shape.
    center = Property(Tuple)

    # The fill color.
    fill_color = ColorTrait

    # The pointer for the 'normal' event state.
    normal_pointer = Pointer("arrow")

    # The pointer for the 'moving' event state.
    moving_pointer = Pointer("hand")

    # The text color.
    text_color = ColorTrait

    # The text displayed in the shape.
    text = Str

    # 'Private' interface ##################################################

    # The difference between the location of a mouse-click and the component's
    # origin.
    _offset_x = Float
    _offset_y = Float

    ###########################################################################
    # 'Interactor' interface
    ###########################################################################

    def normal_key_pressed(self, event):
        """ Event handler. """

        print("normal_key_pressed", event.character)

    def normal_left_down(self, event):
        """ Event handler. """

        if self.is_in(event.x, event.y):
            self.event_state = "moving"

            event.window.mouse_owner = self
            event.window.set_pointer(self.moving_pointer)

            self._offset_x = event.x - self.x
            self._offset_y = event.y - self.y

            # move this shape to the top of the z order. The components are
            # drawn in order, so the last one will be drawn on top
            siblings = self.container.components
            if len(siblings) > 1:
                siblings.remove(self)
                siblings.append(self)

    def moving_mouse_move(self, event):
        """ Event handler. """

        top = event.y + self._offset_y
        bottom = event.y - self._offset_y
        left = event.x - self._offset_x
        right = event.x + self._offset_x

        # Keep the shape fully in the container

        if bottom < 0:
            bottom = 0
        elif top > self.container.height:
            bottom = self.container.height - self.height

        if left < 0:
            left = 0
        elif right > self.container.width:
            left = self.container.width - self.width

        self.position = [left, bottom]
        self.request_redraw()

    def moving_left_up(self, event):
        """ Event handler. """

        self.event_state = "normal"

        event.window.set_pointer(self.normal_pointer)
        event.window.mouse_owner = None

        self.request_redraw()

    def moving_mouse_leave(self, event):
        """ Event handler. """

        self.moving_left_up(event)

    ###########################################################################
    # 'Shape' interface
    ###########################################################################

    def _get_center(self):
        """ Property getter. """

        dx, dy = self.bounds
        ox, oy = self.position

        cx = ox + dx / 2
        cy = oy + dy / 2

        return cx, cy

    def _text_default(self):
        """ Trait initializer. """

        return type(self).__name__

    ###########################################################################
    # Protected 'Shape' interface
    ###########################################################################

    def _distance_between(self, point_1, point_2):
        """ Return the distance between two points. """
        (x1, y1) = point_1
        (x2, y2) = point_2
        return math.sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2))

    def _draw_text(self, gc):
        """ Draw the shape's text. """

        if len(self.text) > 0:
            gc.set_fill_color(self._get_text_color(self.event_state))

            gc.set_font(Font(family=MODERN, size=16))
            tx, ty, tw, th = gc.get_text_extent(self.text)

            dx, dy = self.bounds
            x, y = self.position
            gc.set_text_position(x + (dx - tw) / 2, y + (dy - th) / 2)

            gc.show_text(self.text)

    def _get_fill_color(self, event_state):
        """ Return the fill color based on the event state. """

        if event_state == "normal":
            fill_color = self.fill_color_

        else:
            r, g, b, a = self.fill_color_
            fill_color = (r, g, b, 0.5)

        return fill_color

    def _get_text_color(self, event_state):
        """ Return the text color based on the event state. """

        if event_state == "normal":
            text_color = self.text_color_

        else:
            r, g, b, a = self.text_color_
            text_color = (r, g, b, 0.5)

        return text_color
