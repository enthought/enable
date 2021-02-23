# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
""" Defines the base class for various types of zoom tools.
"""

from numpy import allclose, inf

# Enthought library imports
from traits.api import Enum, Float, HasTraits


class BaseZoomTool(HasTraits):
    """ Defines traits and methods to actually perform the logic of zooming
    onto a plot.
    """

    # If the tool only applies to a particular axis, this attribute is used to
    # determine which range to use.
    axis = Enum("x", "y")

    # The maximum ratio between the original data space bounds and the
    # zoomed-in data space bounds.
    # If None, then there is no limit (not advisable!).
    max_zoom_in_factor = Float(1e5, allow_none=True)

    # The maximum ratio between the zoomed-out data space bounds and the
    # original bounds.  If None, then there is no limit.
    max_zoom_out_factor = Float(1e5, allow_none=True)

    def _zoom_limit_reached(self, orig_position, orig_bounds, new_position,
                            new_bounds):
        """ Returns True if the new low and high exceed the maximum zoom
        limits
        """
        if orig_bounds == inf:
            # There isn't really a good way to handle the case when the
            # original bounds were infinite, since any finite zoom
            # range will certainly exceed whatever zoom factor is set.
            # In this case, we just allow unbounded levels of zoom.
            return False

        if allclose(orig_bounds, 0.0):
            return True
        if allclose(new_bounds, 0.0):
            return True
        if ((new_bounds / orig_bounds) > self.max_zoom_out_factor
                or (orig_bounds / new_bounds) > self.max_zoom_in_factor):
            return True
        return False

    # ------------------------------------------------------------------------
    # Utility methods for computing axes, coordinates, etc.
    # ------------------------------------------------------------------------

    def _get_range_index(self):
        """ Returns the index into the view_position and view_bounds
            depending on value of self.axis.
        """
        if self.axis == "x":
            return 0
        else:
            return 1

    def _get_axis_coord(self, event, axis="x"):
        """ Returns the coordinate of the event along the axis of interest
        to the tool (or along the orthogonal axis, if axis="value").
        """
        event_pos = (event.x, event.y)
        if axis == "x":
            return event_pos[self._determine_axis()]
        else:
            return event_pos[1 - self._determine_axis()]

    def _determine_axis(self):
        """ Determines whether the index of the coordinate along the axis of
        interest is the first or second element of an (x,y) coordinate tuple.
        """
        if self.axis == "x":
            return 0
        else:
            return 1
