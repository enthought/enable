# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!

"""A drag drawn box
"""

# Standard library imports.

# Major packages.

# Enthought library imports
from enable.primitives.api import Box
from enable.enable_traits import Pointer
from traits.api import Event, Float, Trait, Tuple

# Application specific imports.

# Local imports.

##############################################################################
# class 'DragBox'
##############################################################################


class DragBox(Box):
    """A drag drawn box
    """

    ##########################################################################
    # Traits
    ##########################################################################

    # 'DragBox' interface ############################################

    # Event fired when complete:
    complete = Event

    # Constraints on size:
    x_bounds = Trait(None, None, Tuple(Float, Float))
    y_bounds = Trait(None, None, Tuple(Float, Float))

    # Pointers. ####

    # Pointer for the complete state:
    complete_pointer = Pointer("cross")

    # Pointer for the drawing state:
    drawing_pointer = Pointer("cross")

    # Pointer for the normal state:
    normal_pointer = Pointer("cross")

    # Private traits

    # Position of the left down:
    start_x = Float
    start_y = Float

    ##########################################################################
    # 'object' interface
    ##########################################################################

    ##########################################################################
    # 'Component' interface
    ##########################################################################

    # 'normal' state ######################################################

    def normal_left_down(self, event):
        """ Handle the left button down in the 'normal' state. """

        self.event_state = "drawing"
        self.pointer = self.drawing_pointer

        self.start_x = event.x
        self.start_y = event.y
        self._set_bounds(event)

    def normal_mouse_move(self, event):
        """ Handle the mouse moving in the 'normal' state. """

        self.pointer = self.normal_pointer

    # 'drawing' state #####################################################

    def drawing_mouse_move(self, event):
        """ Handle the mouse moving in the 'drawing' state. """

        self._set_bounds(event)

    def drawing_left_up(self, event):
        """ Handle the left mouse button coming up in the 'drawing' state. """

        self.event_state = "complete"
        self.pointer = self.complete_pointer

        self.complete = True
        self._set_bounds(event)

    ##########################################################################
    # Private interface
    ##########################################################################

    def _set_bounds(self, event):
        """
        Sets the bounds based on start_x, start_y, the event position
        and any constrants.
        """
        if self.x_bounds is not None:
            x, dx = self.x_bounds
        else:
            x = min(self.start_x, event.x)
            dx = abs(event.x - self.start_x)
        if self.y_bounds is not None:
            y, dy = self.y_bounds
        else:
            y = min(self.start_y, event.y)
            dy = abs(event.y - self.start_y)
        self.bounds = (x, y, dx, dy)
