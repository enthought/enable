# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
""" A moveable box shape. """
from enable.primitives.shape import Shape


class Box(Shape):
    """ A moveable box shape. """

    # Protected 'Component' interface.
    # ---------------------------------

    def _draw_mainlayer(self, gc, view_bounds=None, mode="default"):
        """ Draw the component. """
        with gc:
            gc.set_fill_color(self._get_fill_color(self.event_state))

            dx, dy = self.bounds
            x, y = self.position
            gc.rect(x, y, dx, dy)
            gc.fill_path()

            # Draw the shape's text.
            self._draw_text(gc)
