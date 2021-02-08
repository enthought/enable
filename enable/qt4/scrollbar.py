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
the standard Qt one.
"""

from pyface.qt import QtCore, QtGui
from traits.api import Any, Bool, Enum, Float, Int, Property, Trait, TraitError

from enable.component import Component


def valid_range(object, name, value):
    """ Verify that a set of range values for a scrollbar is valid.
    """
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
    """ Verify that a specified scroll bar position is valid.
    """
    try:
        low, high, page_size, line_size = object.range
        x = max(min(float(value), high - page_size), low)
        return x
    except Exception:
        raise
    raise TraitError


class QResizableScrollBar(QtGui.QScrollBar):

    resized = QtCore.Signal()

    def resizeEvent(self, event):
        super(QResizableScrollBar, self).resizeEvent(event)
        self.resized.emit()


class NativeScrollBar(Component):
    "An Enable scrollbar component that wraps/embeds the native Qt scrollbar"

    # ------------------------------------------------------------------------
    # Public Traits
    # ------------------------------------------------------------------------

    # The current position of the scroll bar.  This must be within the range
    # (self.low, self.high)
    scroll_position = Trait(0.0, valid_scroll_position)

    # A tuple (low, high, page_size, line_size).  Can be accessed using
    # convenience properties (see below).
    range = Trait((0.0, 100.0, 10.0, 1.0), valid_range)

    # The orientation of the scrollbar
    orientation = Trait("horizontal", "vertical")

    # Is y=0 at the top or bottom?
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
    _control = Any
    _clean = Bool(False)
    _last_widget_x = Float(0)
    _last_widget_y = Float(0)
    _last_widget_height = Float(0)
    _last_widget_width = Float(0)

    # Indicates whether or not the widget needs to be re-drawn after being
    # repositioned and resized
    _widget_moved = Bool(True)

    # Set to True if something else has updated the scroll position and
    # the widget needs to redraw.  This is not set to True if the widget
    # gets updated via user mouse interaction, since Qt is then responsible
    # for updating the scrollbar.
    _scroll_updated = Bool(True)

    def destroy(self):
        """ Destroy the native widget associated with this component.
        """
        if self._control is not None:
            self._control.hide()
            self._control.deleteLater()
            self._control = None

    def __del__(self):
        # Pray that we do not participate in a cycle.
        self.destroy()

    def _get_abs_coords(self, x, y):
        return self.container.get_absolute_coords(x, y)

    def _draw_mainlayer(self, gc, view_bounds=None, mode="default"):
        x_pos, y_pos = self.position
        x_size, y_size = map(int, self.bounds)

        qt_xpos, qt_ypos = self._get_abs_coords(x_pos, y_pos + y_size - 1)

        # We have to do this flip_y business because Qt and enable use opposite
        # coordinate systems, and enable defines the component's position as
        # its lower left corner, while Qt defines it as the upper left corner.
        window = self.window
        if window is None:
            return
        qt_ypos = window._flip_y(qt_ypos)
        qt_xpos = int(qt_xpos)
        qt_ypos = int(qt_ypos)

        if not self._control:
            self._create_control(window, self.range, self.scroll_position)

        if self._widget_moved:
            if (self._last_widget_x != qt_xpos
                    or self._last_widget_y != qt_ypos):
                self._control.move(qt_xpos, qt_ypos)
            controlsize = self._control.size()
            if x_size != controlsize.width() or y_size != controlsize.height():
                self._control.resize(x_size, y_size)

        if self._scroll_updated:
            self._update_control(self.range, self.scroll_position)

        # self._control.raise_()

        self._last_widget_x = qt_xpos
        self._last_widget_y = qt_ypos
        self._last_widget_width = x_size
        self._last_widget_height = y_size
        self._scroll_updated = False
        self._widget_moved = False

    def _create_control(self, window, enable_range, value):
        qt_orientation = {
            'horizontal': QtCore.Qt.Horizontal,
            'vertical': QtCore.Qt.Vertical
        }[self.orientation]

        self._control = QResizableScrollBar(qt_orientation, window.control)
        self._update_control(enable_range, value)
        self._control.valueChanged.connect(self._update_enable_pos)
        self._control.sliderPressed.connect(self._on_slider_pressed)
        self._control.sliderReleased.connect(self._on_slider_released)
        self._control.resized.connect(self._control_resized)
        self._control.destroyed.connect(self._on_destroyed)
        self._control.setVisible(True)

    def _update_control(self, enable_range, value):
        minimum, maximum, page_size, line_size = enable_range
        # The maximum value of a QScrollBar is the maximum position of the
        # scroll bar, not the document length. We need to subtract the length
        # of the scroll bar itself.
        max_value = maximum - page_size
        # invert values for vertical ranges because of coordinate system issues
        value = self._correct_value(value, minimum, max_value)

        self._control.setMinimum(minimum)
        self._control.setMaximum(max_value)
        self._control.setValue(value)
        self._control.setPageStep(page_size)
        self._control.setSingleStep(line_size)

    def _correct_value(self, value, min_value, max_value):
        """ Correct vertical position values for Qt and Enable conventions

        Enable expects vertical scroll_position to be measured with origin at
        the bottom and positive going upwards, while Qt scrollbar values are
        measured with origin at the top and positive going down.

        Parameters
        ----------
        value : float
            The position value in either Enable or Qt conventions.
        max_value : float
            The maximum value that the Qt scrollbar can be set to (height of
            the scrolled component, less the page size).
        """
        if self.orientation != "vertical":
            return value
        return max_value - (value - min_value)

    # ------------------------------------------------------------------------
    # Qt Event handlers
    # ------------------------------------------------------------------------

    def _update_enable_pos(self, value):
        # invert values for vertical ranges because of coordinate system issues
        value = self._correct_value(
            value, self.low, self.high - self.page_size
        )
        self.scroll_position = value

    def _on_slider_pressed(self):
        self.mouse_thumb = "down"

    def _on_slider_released(self):
        self.mouse_thumb = "up"

    def _control_resized(self):
        self._widget_moved = True
        self.request_redraw()

    def _on_destroyed(self):
        self._control = None

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
        # FIXME: convert to Qt.
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
        self._clean = False
        self.range = (low, high, page_size, line_size)

    def _get_high(self):
        return self.range[1]

    def _set_high(self, high):
        low, ignore, page_size, line_size = self.range
        self._clean = False
        self.range = (low, high, page_size, line_size)

    def _get_page_size(self):
        return self.range[2]

    def _set_page_size(self, page_size):
        low, high, ignore, line_size = self.range
        self._clean = False
        self.range = (low, high, page_size, line_size)

    def _get_line_size(self):
        return self.range[3]

    def _set_line_size(self, line_size):
        low, high, page_size, ignore = self.range
        self._clean = False
        self.range = (low, high, page_size, line_size)
