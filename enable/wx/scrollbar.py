# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
"""
Define a standard horizontal and vertical Enable scrollbar component that wraps
the standard WX one.
"""

# Major library imports
import wx

# Enthought Imports
from traits.api import Property, Trait, TraitError, Any, Enum, Bool, Int

from enable.component import Component


def valid_range(object, name, value):
    "Verify that a set of range values for a scrollbar is valid"
    try:
        if (type(value) in (tuple, list)) and (len(value) == 4):
            low, high, page_size, line_size = value
            if high < low:
                low, high = high, low
            elif high == low:
                high = low + 1.0
            page_size = max(min(page_size, high - low), 0.0)
            line_size = max(min(line_size, page_size), 0.0)
            return (
                float(low),
                float(high),
                float(page_size),
                float(line_size),
            )
    except Exception:
        raise
    raise TraitError


valid_range.info = "a (low,high,page_size,line_size) range tuple"


def valid_scroll_position(object, name, value):
    "Verify that a specified scroll bar position is valid"
    try:
        low, high, page_size, line_size = object.range
        if value > high - page_size:
            value = high - page_size
        elif value < low:
            value = low
        return value
    except Exception:
        raise
    raise TraitError


class NativeScrollBar(Component):
    "An Enable scrollbar component that wraps/embeds the standard WX scrollbar"

    # ------------------------------------------------------------------------
    # Public Traits
    # ------------------------------------------------------------------------

    # The current position of the scroll bar.  This must be within the range
    # (self.low, self.high)
    scroll_position = Trait(0.0, valid_scroll_position)

    # A tuple (low, high, page_size, line_size).  Can be accessed using
    # convenience properties (see below).  Low and High refer to the conceptual
    # bounds of the region represented by the full scroll bar.  Note that
    # the maximum value of scroll_position is actually (high - page_size), and
    # not just the value of high.
    range = Trait((0.0, 100.0, 10.0, 1.0), valid_range)

    # The orientation of the scrollbar
    orientation = Trait("horizontal", "vertical")

    # The location of y=0
    origin = Trait("bottom", "top")

    # Determines if the scroll bar should be visible and respond to events
    enabled = Bool(True)

    # The scroll increment associated with a single mouse wheel increment
    mouse_wheel_speed = Int(3)

    # Expose scroll_position, low, high, page_size as properties
    low = Property
    high = Property
    page_size = Property
    line_size = Property

    # This represents the state of the mouse button on the scrollbar thumb.
    # External classes can monitor this to detect when the user starts and
    # finishes interacting with this scrollbar via the scrollbar thumb.
    mouse_thumb = Enum("up", "down")

    # ------------------------------------------------------------------------
    # Private Traits
    # ------------------------------------------------------------------------
    _control = Any(None)
    _last_widget_x = Int(0)
    _last_widget_y = Int(0)
    _last_widget_height = Int(0)
    _last_widget_width = Int(0)

    # Indicates whether or not the widget needs to be re-drawn after being
    # repositioned and resized
    _widget_moved = Bool(True)

    # Set to True if something else has updated the scroll position and
    # the widget needs to redraw.  This is not set to True if the widget
    # gets updated via user mouse interaction, since WX is then responsible
    # for updating the scrollbar.
    _scroll_updated = Bool(True)

    # ------------------------------------------------------------------------
    # Public Methods
    # ------------------------------------------------------------------------

    def destroy(self):
        """ Destroy the native widget associated with this component.
        """
        if self._control:
            self._control.Destroy()

    # ------------------------------------------------------------------------
    # Protected methods
    # ------------------------------------------------------------------------

    def __del__(self):
        self.destroy()

    def _get_abs_coords(self, x, y):
        return self.container.get_absolute_coords(x, y)

    def _draw_mainlayer(self, gc, view_bounds=None, mode="default"):
        x_pos, y_pos = self.position
        x_size, y_size = self.bounds

        wx_xpos, wx_ypos = self._get_abs_coords(x_pos, y_pos + y_size - 1)

        # We have to do this flip_y business because wx and enable use opposite
        # coordinate systems, and enable defines the component's position as
        # its lower left corner, while wx defines it as the upper left corner.
        window = getattr(gc, "window", None)
        if window is None:
            return
        wx_ypos = window._flip_y(wx_ypos)

        low, high, page_size, line_size = self.range
        wxpos, wxthumbsize, wxrange = self._enable_to_wx_spec(
            self.range + (self.scroll_position,)
        )

        if not self._control:
            self._create_control(window, wxpos, wxthumbsize, wxrange)

        if self._widget_moved:
            if (self._last_widget_x != wx_xpos
                    or self._last_widget_y != wx_ypos):
                self._control.SetPosition(wx.Point(wx_xpos, wx_ypos))
            controlsize = self._control.GetSize()
            if x_size != controlsize[0] or y_size != controlsize[1]:
                self._control.SetSize(wx.Size(x_size, y_size))

        if self._scroll_updated:
            self._control.SetScrollbar(
                wxpos, wxthumbsize, wxrange, wxthumbsize, True
            )

        self._last_widget_x = int(wx_xpos)
        self._last_widget_y = int(wx_ypos)
        self._last_widget_width = int(x_size)
        self._last_widget_height = int(y_size)
        self._scroll_updated = False
        self._widget_moved = False

    def _create_control(self, window, wxpos, wxthumbsize, wxrange):
        if self.orientation == "horizontal":
            wxstyle = wx.HORIZONTAL
        else:
            wxstyle = wx.VERTICAL
        self._control = wx.ScrollBar(window.control, style=wxstyle)
        self._control.SetScrollbar(
            wxpos, wxthumbsize, wxrange, wxthumbsize, True
        )
        wx.EVT_SCROLL(self._control, self._wx_scroll_handler)
        wx.EVT_SET_FOCUS(self._control, self._yield_focus)
        wx.EVT_SCROLL_THUMBTRACK(self._control, self._thumbtrack)
        wx.EVT_SCROLL_THUMBRELEASE(self._control, self._thumbreleased)
        wx.EVT_SIZE(self._control, self._control_resized)

    # ------------------------------------------------------------------------
    # WX Event handlers
    # ------------------------------------------------------------------------

    def _thumbtrack(self, event):
        self.mouse_thumb = "down"
        self._wx_scroll_handler(event)

    def _thumbreleased(self, event):
        self.mouse_thumb = "up"
        self._wx_scroll_handler(event)

    def _control_resized(self, event):
        self._widget_moved = True
        self.request_redraw()

    def _yield_focus(self, event):
        """
        Yields focus to our window, when we acquire focus via user interaction.
        """
        window = event.GetWindow()
        if window:
            window.SetFocus()

    def _wx_scroll_handler(self, event):
        """Handle wx scroll events"""
        # If the user moved the scrollbar, set the scroll position, but don't
        # tell wx to move the scrollbar.  Doing so causes jerkiness
        self.scroll_position = self._wx_to_enable_pos(
            self._control.GetThumbPosition()
        )

    def _enable_to_wx_spec(self, enable_spec):
        """
        Return the WX equivalent of an enable scroll bar specification from
        a tuple of (low, high, page_size, line_size, position).
        Returns (position, thumbsize, range)
        """
        low, high, page_size, line_size, position = enable_spec
        if self.origin == "bottom" and self.orientation == "vertical":
            position = (high - page_size) - position + 1
        if line_size == 0.0:
            return (0, high - low, high - low)
        else:
            return [
                int(round(x))
                for x in (
                    (position - low) / line_size,
                    page_size / line_size,
                    (high - low) / line_size,
                )
            ]

    def _wx_to_enable_pos(self, pos):
        """
        Translate the position that the Wx scrollbar returns into the position
        we store internally.  The difference is that we have a high and a low
        and a line size, while wx assumes low is 0 and line size is 1.
        """
        low, high, page_size, line_size = self.range
        enablepos = pos * line_size
        # If we're a veritcal scrollbar with a bottom origin, flip
        # the coordinates, since in WX the origin is always the top.
        if self.origin == "bottom" and self.orientation == "vertical":
            enablepos = (high - low - page_size) - enablepos
        enablepos += low
        return enablepos

    # ------------------------------------------------------------------------
    # Basic trait event handlers
    # ------------------------------------------------------------------------

    def _range_changed(self):
        low, high, page_size, line_size = self.range
        self.scroll_position = max(
            min(self.scroll_position, high - page_size), low
        )
        self._scroll_updated = True
        self.request_redraw()

    def _range_items_changed(self):
        self._range_changed()

    def _mouse_wheel_changed(self, event):
        event.handled = True
        self.scroll_position += (
            event.mouse_wheel * self.range[3] * self.mouse_wheel_speed
        )

    def _scroll_position_changed(self):
        self._scroll_updated = True
        self.request_redraw()

    def _bounds_changed(self, old, new):
        super(NativeScrollBar, self)._bounds_changed(old, new)
        self._widget_moved = True
        self.request_redraw()

    def _bounds_items_changed(self, event):
        super(NativeScrollBar, self)._bounds_items_changed(event)
        self._widget_moved = True
        self.request_redraw()

    def _position_changed(self, old, new):
        super(NativeScrollBar, self)._position_changed(old, new)
        self._widget_moved = True
        self.request_redraw()

    def _position_items_changed(self, event):
        super(NativeScrollBar, self)._position_items_changed(event)
        self._widget_moved = True
        self.request_redraw()

    # ------------------------------------------------------------------------
    # Property getters and setters
    # ------------------------------------------------------------------------

    def _get_low(self):
        return self.range[0]

    def _set_low(self, low):
        ignore, high, page_size, line_size = self.range
        self._scroll_updated = True
        self.range = (low, high, page_size, line_size)

    def _get_high(self):
        return self.range[1]

    def _set_high(self, high):
        low, ignore, page_size, line_size = self.range
        self._scroll_updated = True
        self.range = (low, high, page_size, line_size)

    def _get_page_size(self):
        return self.range[2]

    def _set_page_size(self, page_size):
        low, high, ignore, line_size = self.range
        self._scroll_updated = True
        self.range = (low, high, page_size, line_size)

    def _get_line_size(self):
        return self.range[3]

    def _set_line_size(self, line_size):
        low, high, page_size, ignore = self.range
        self._scroll_updated = True
        self.range = (low, high, page_size, line_size)
