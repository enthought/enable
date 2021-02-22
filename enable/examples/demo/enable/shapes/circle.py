# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
""" A moveable circle shape. """
from traits.api import Float
from enable.primitives.shape import Shape


class Circle(Shape):
    """ A moveable circle shape. """

    # The radius of the circle.
    radius = Float

    # 'CoordinateBox' interface.
    # ---------------------------

    def _bounds_changed(self):
        """ Static trait change handler. """
        w, h = self.bounds
        self.radius = min(w, h) / 2.0

    # 'Component' interface.
    # -----------------------

    def is_in(self, x, y):
        """ Return True if a point is considered to be 'in' the component. """
        return self._distance_between(self.center, (x, y)) <= self.radius

    # Protected 'Component' interface.
    # ---------------------------------

    def _draw_mainlayer(self, gc, view_bounds=None, mode="default"):
        """ Draw the component. """
        with gc:
            gc.set_fill_color(self._get_fill_color(self.event_state))

            x, y = self.position
            gc.arc(
                x + self.radius,
                y + self.radius,
                self.radius,
                0,
                2 * 3.14159,
                False,
            )
            gc.fill_path()

            # Draw the shape's text.
            self._draw_text(gc)

    # 'Circle' interface.
    # --------------------

    def _radius_changed(self):
        """ Static trait change handler. """
        diameter = self.radius * 2
        self.bounds = [diameter, diameter]
