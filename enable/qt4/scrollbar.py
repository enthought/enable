"""
Define a standard horizontal and vertical Enable scrollbar component that wraps
the standard Qt one.
"""

# Major library imports
from types import ListType, TupleType

# Enthought library imports
from traits.api import Any, Bool, Float, Int, Property, Trait, \
    TraitError

from enable.component import Component


def valid_range(object, name, value):
    "Verify that a set of range values for a scrollbar is valid"
    try:
        if (type(value) in (TupleType, ListType)) and (len(value) == 4):
            low, high, page_size, line_size = value
            if high < low:
                low, high = high, low
            elif high == low:
                high = low + 1.0
            page_size = max(min(page_size, high - low), 0.0)
            line_size = max(min(line_size, page_size), 0.0)
            return (float(low), float(high), float(page_size), float(line_size))
    except:
        raise
    raise TraitError

valid_range.info = 'a (low,high,page_size,line_size) range tuple'


def valid_scroll_position(object, name, value):
    "Verify that a specified scroll bar position is valid"
    try:
        low, high, page_size, line_size = object.range
        return max(min(float(value), high - page_size), low)
    except:
        raise
    raise TraitError


class NativeScrollBar(Component):
    "An Enable scrollbar component that wraps/embeds the native Qt scrollbar"

    #------------------------------------------------------------------------
    # Public Traits
    #------------------------------------------------------------------------

    # The current position of the scroll bar.  This must be within the range
    # (self.low, self.high)
    scroll_position = Trait( 0.0, valid_scroll_position )

    # A tuple (low, high, page_size, line_size).  Can be accessed using
    # convenience properties (see below).
    range = Trait( ( 0.0, 100.0, 10.0, 1.0 ), valid_range )

    # The orientation of the scrollbar
    orientation = Trait("horizontal", "vertical")

    # Is y=0 at the top or bottom?
    origin = Trait('bottom', 'top')

    # Determines if the scroll bar should be visible and respond to events
    enabled = Bool(True)

    # The scroll increment associated with a single mouse wheel increment
    mouse_wheel_speed = Int(3)

    # Expose scroll_position, low, high, page_size as properties
    low = Property
    high = Property
    page_size = Property
    line_size = Property

    #------------------------------------------------------------------------
    # Private Traits
    #------------------------------------------------------------------------
    _control = Any
    _clean = Bool(False)
    _last_widget_x = Float(0)
    _last_widget_y = Float(0)
    _last_widget_height = Float(0)
    _list_widget_width = Float(0)

    #------------------------------------------------------------------------
    # Property getters and setters
    #------------------------------------------------------------------------

    def _get_low(self):
        return self.range[0]

    def _set_low(self, low):
        ignore, high, page_size, line_size = self.range
        self._clean = False
        self.range =(low, high, page_size, line_size)

    def _get_high(self):
        return self.range[1]

    def _set_high(self, high):
        low, ignore, page_size, line_size = self.range
        self._clean = False
        self.range =(low, high, page_size, line_size)

    def _get_page_size(self):
        return self.range[2]

    def _set_page_size(self, page_size):
        low, high, ignore, line_size = self.range
        self._clean = False
        self.range =(low, high, page_size, line_size)

    def _get_line_size(self):
        return self.range[3]

    def _set_line_size(self, line_size):
        low, high, page_size, ignore = self.range
        self._clean = False
        self.range =(low, high, page_size, line_size)

