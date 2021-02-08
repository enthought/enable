# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
""" LineState Class

The LineState class is used by the GraphicsState for Kiva backends which need
to have their state tracked by Python, rather than by an internal graphics
state (eg. Wx, SVG and PDF backends, but not Agg or QPainter).
"""

from numpy import alltrue, array, asarray, shape, sometrue

from .constants import NO_DASH


def exactly_equal(arr1, arr2):
    return shape(arr1) == shape(arr2) and alltrue(arr1 == arr2)


def is_dashed(dash):
    # if all the values in the dash settings are 0, then it is a solid line.
    result = 0
    if dash is not None and sometrue(asarray(dash[1]) != 0):
        result = 1
    return result


def line_state_equal(line1, line2):
    """ Compares two `LineState` objects to see if they are equivalent.

        This is generally called by device-specific drawing routines
        before they stroke a path. It determines whether previously set
        line settings are equivalent to desired line settings for this
        drawing command.  If true, the routine can bypass all the
        work needed to set all the line settings of the graphics device.

        With the current Python implementation, this may not provide any
        time savings over just setting all the graphics state values.
        However, in C this could be a very fast memcmp if the C structure
        is set up correctly.

        While this could be the __cmp__ method for `LineState`, I have
        left it as a function because I think it will move to C and be
        used to compare structures.
    """
    # ---------------------------------------------------------------------
    # line_dash is a little persnickety.  It is a 2-tuple
    # with the second entry being an array.  If the arrays are different,
    # just comparing the tuple will yield true because of how rich
    # the result from the array comparison is a non-empty array which
    # tests true.  Thus, the tuple comparison will test true even if the
    # arrays are different.  Its almost like we need a "deep compare"
    # method or something like that.
    #
    # Note: I think should be easy, but is breaking because of a bug in
    #       Numeric.  Waiting for confirmation.
    # ---------------------------------------------------------------------
    dash_equal = line1.line_dash[0] == line2.line_dash[0] and exactly_equal(
        line1.line_dash[1], line2.line_dash[1]
    )
    result = (
        dash_equal
        and exactly_equal(line1.line_color, line2.line_color)
        and line1.line_width == line2.line_width
        and line1.line_cap == line2.line_cap
        and line1.line_join == line2.line_join
    )
    return result


class LineState(object):
    """ Stores information about the current line drawing settings.

    This is split off from `GraphicsState` to make it easier to
    track line state changes.  All the methods for setting
    these variables are left in the GraphicsStateBase class.
    """

    def __init__(self, color, width, cap, join, dash):
        """ Creates a new `LineState` object.

        All input arguments that are containers are copied
        by the constructor.  This prevents two `LineState` objects
        from ever sharing and modifying the other's data.
        """
        self.line_color = array(color, copy=1)
        self.line_width = width
        self.line_cap = cap
        self.line_join = join
        if not dash:
            # always set line_dash to be a tuple
            self.line_dash = NO_DASH
        else:
            self.line_dash = (dash[0], array(dash[1], copy=1))

    def copy(self):
        """ Makes a copy of the current line state
        """
        # Could just use deepcopy...
        return LineState(
            self.line_color,
            self.line_width,
            self.line_cap,
            self.line_join,
            self.line_dash,
        )

    def is_dashed(self):
        # if line_dash only has one entry, it is a solid line.
        return is_dashed(self.line_dash)
