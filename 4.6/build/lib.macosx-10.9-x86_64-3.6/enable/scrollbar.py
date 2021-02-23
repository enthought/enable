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
Define a standard horizontal and vertical Enable 'scrollbar' component.

The scrollbar uses images for the pieces of the scrollbar itself and stretches
them appropriately in the draw phase.
"""

from functools import reduce

# PZW: Define a scrollbar that uses the system/wx-native scrollbar instead
# of drawing our own.

from traits.api import Event, Property, Trait, TraitError
from traitsui.api import Group, View

# Relative imports
from .component import Component
from .enable_traits import layout_style_trait


# -----------------------------------------------------------------------------
# Constants:
# -----------------------------------------------------------------------------

# Scroll bar zones:
NO_SCROLL = -1
LINE_UP = 0
PAGE_UP = 1
LINE_DOWN = 2
PAGE_DOWN = 3
SLIDER = 4

# Scrollbar suffix names by zone:
zone_suffixes = [
    "_line_up_suffix",
    "",
    "_line_down_suffix",
    "",
    "_slider_suffix",
]

# Scroll information look-up table:
scroll_info = [
    ("line_up", 1.0, 3),
    ("page_up", 1.0, 2),
    ("line_down", -1.0, 3),
    ("page_down", -1.0, 2),
]

# Scroll bar images and sizes:
sb_image = {}
v_width = 0
v_height = 0
vs_height = 0
h_width = 0
h_height = 0
hs_width = 0


# -----------------------------------------------------------------------------
# Traits validators
# -----------------------------------------------------------------------------


def valid_range(object, name, value):
    "Verify that a set of range values for a scrollbar is valid"
    try:
        if (type(value) is tuple) and (len(value) == 4):
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
        pass
    raise TraitError


valid_range.info = "a (low,high,page_size,line_size) range tuple"


def valid_position(object, name, value):
    "Verify that a specified scroll bar position is valid"
    try:
        low, high, page_size, line_size = object.range
        return max(min(float(value), high - page_size), low)
    except Exception:
        pass
    raise TraitError


class ScrollBar(Component):

    position = Trait(0.0, valid_position)
    range = Trait((0.0, 100.0, 10.0, 1.0), valid_range)
    style = layout_style_trait

    line_up = Event
    line_down = Event
    page_up = Event
    page_down = Event
    scroll_done = Event

    traits_view = View(
        Group("<component>", id="component"),
        Group("<links>", id="links"),
        Group(
            "style",
            " ",
            "position",
            "low",
            "high",
            "page_size",
            "line_size",
            id="scrollbar",
            style="custom",
        ),
    )

    # -------------------------------------------------------------------------
    #  Property definitions:
    # -------------------------------------------------------------------------

    def __low_get(self):
        return self.range[0]

    def __low_set(self, low):
        ignore, high, page_size, line_size = self.range
        self.range = (low, high, page_size, line_size)

    def __high_get(self):
        return self.range[1]

    def __high_set(self, high):
        low, ignore, page_size, line_size = self.range
        self.range = (low, high, page_size, line_size)

    def __page_size_get(self):
        return self.range[2]

    def __page_size_set(self, page_size):
        low, high, ignore, line_size = self.range
        self.range = (low, high, page_size, line_size)

    def __line_size_get(self):
        return self.range[3]

    def __line_size_set(self, line_size):
        low, high, page_size, ignore = self.range
        self.range = (low, high, page_size, line_size)

    # Define 'position, low, high, page_size' properties:
    low = Property(__low_get, __low_set)
    high = Property(__high_get, __high_set)
    page_size = Property(__page_size_get, __page_size_set)
    line_size = Property(__line_size_get, __line_size_set)

    def __init__(self, **traits):
        if v_width == 0:
            self._init_images()
        Component.__init__(self, **traits)
        self._scrolling = self._zone = NO_SCROLL
        self._line_up_suffix = (
            self._line_down_suffix
        ) = self._slider_suffix = ""
        self._style_changed(self.style)

    def _init_images(self):
        "One time initialization of the scrollbar images"
        global sb_image, v_width, v_height, vs_height
        global h_width, h_height, hs_width

        for name in [
            "aup",
            "adown",
            "aleft",
            "aright",
            "vtop",
            "vbottom",
            "vmid",
            "vpad",
            "hleft",
            "hright",
            "hmid",
            "hpad",
        ]:
            sb_image[name] = self.image_for("=sb_%s" % name)
            sb_image[name + "_over"] = self.image_for("=sb_%s_over" % name)
            sb_image[name + "_down"] = self.image_for("=sb_%s_down" % name)
        sb_image["vtrack"] = self.image_for("=sb_vtrack")
        sb_image["htrack"] = self.image_for("=sb_htrack")
        v_width = sb_image["vtrack"].width()
        vs_height = reduce(
            lambda a, b: a + sb_image[b].height(),
            ["vtop", "vbottom", "vmid"],
            0,
        )
        v_height = reduce(
            lambda a, b: a + sb_image[b].height(), ["aup", "adown"], vs_height
        )
        hs_width = reduce(
            lambda a, b: a + sb_image[b].width(),
            ["hleft", "hright", "hmid"],
            0,
        )
        h_width = reduce(
            lambda a, b: a + sb_image[b].width(), ["aleft", "aright"], hs_width
        )
        h_height = sb_image["htrack"].height()

    def _range_changed(self):
        "Handle any of the range elements values being changed"
        low, high, page_size, line_size = self.range
        self.position = max(min(self.position, high - page_size), low)
        self.redraw()

    def _position_changed(self):
        self.redraw()

    def _style_changed(self, style):
        "Handle the orientation style being changed"
        if style[0] == "v":
            self.min_width = self.max_width = v_width
            self.min_height = v_height
            self.max_height = 99999.0
            self.stretch_width = 0.0
            self.stretch_height = 1.0
        else:
            self.min_width = h_width
            self.max_width = 99999.0
            self.min_height = self.max_height = h_height
            self.stretch_width = 1.0
            self.stretch_height = 0.0

    def _draw(self, gc):
        "Draw the contents of the control"
        with gc:
            if self.style[0] == "v":
                self._draw_vertical(gc)
            else:
                self._draw_horizontal(gc)

    def _draw_vertical(self, gc):
        "Draw a vertical scrollbar"
        low, high, page_size, line_size = self.range
        position = self.position
        x, y, dx, dy = self.bounds
        adown = sb_image["adown" + self._line_down_suffix]
        adown_dy = adown.height()
        aup = sb_image["aup" + self._line_up_suffix]
        aup_dy = aup.height()
        vtrack = sb_image["vtrack"]
        t_dy = dy - aup_dy - adown_dy
        t_y = s_y = y + adown_dy
        u_y = y + dy - aup_dy
        gc.stretch_draw(vtrack, x, t_y, dx, t_dy)
        gc.draw_image(adown, (x, y, dx, adown_dy))
        gc.draw_image(aup, (x, u_y, dx, aup_dy))
        if page_size > 0.0:
            s_dy = max(vs_height, round((page_size * t_dy) / (high - low)))
            self._range = range_dy = t_dy - s_dy
            range = high - low - page_size
            if range > 0.0:
                s_y = round(s_y + (((position - low) * range_dy) / range))
            suffix = self._slider_suffix
            vbottom = sb_image["vbottom" + suffix]
            vbottom_dy = vbottom.height()
            vtop = sb_image["vtop" + suffix]
            vtop_dy = vtop.height()
            vmid = sb_image["vmid" + suffix]
            vmid_dy = vmid.height()
            gc.stretch_draw(
                sb_image["vpad" + suffix],
                x,
                s_y + vbottom_dy,
                dx,
                s_dy - vbottom_dy - vtop_dy,
            )
            gc.draw_image(vbottom, (x, s_y, dx, vbottom_dy))
            gc.draw_image(vtop, (x, s_y + s_dy - vtop_dy, dx, vtop_dy))
            gc.draw_image(
                vmid,
                (
                    x,
                    round(
                        s_y
                        + vbottom_dy
                        + (s_dy - vbottom_dy - vtop_dy - vmid_dy) / 2.0
                    ),
                    dx,
                    vmid_dy,
                ),
            )
            self._info = (t_y, s_y, s_y + s_dy, u_y)

    def _draw_horizontal(self, gc):
        "Draw a horizontal scroll bar"
        low, high, page_size, line_size = self.range
        position = self.position
        x, y, dx, dy = self.bounds
        aleft = sb_image["aleft" + self._line_up_suffix]
        aleft_dx = aleft.width()
        aright = sb_image["aright" + self._line_down_suffix]
        aright_dx = aright.width()
        htrack = sb_image["htrack"]
        t_dx = dx - aleft_dx - aright_dx
        t_x = s_x = x + aleft_dx
        r_x = x + dx - aright_dx
        gc.stretch_draw(htrack, t_x, y, t_dx, dy)
        gc.draw_image(aleft, (x, y, aleft_dx, dy))
        gc.draw_image(aright, (r_x, y, aright_dx, dy))
        if page_size > 0.0:
            s_dx = max(hs_width, round((page_size * t_dx) / (high - low)))
            self._range = range_dx = t_dx - s_dx
            range = high - low - page_size
            if range > 0.0:
                s_x = round(s_x + (((position - low) * range_dx) / range))
            suffix = self._slider_suffix
            hleft = sb_image["hleft" + suffix]
            hleft_dx = hleft.width()
            hright = sb_image["hright" + suffix]
            hright_dx = hright.width()
            hmid = sb_image["hmid" + suffix]
            hmid_dx = hmid.width()
            gc.stretch_draw(
                sb_image["hpad" + suffix],
                s_x + hleft_dx,
                y,
                s_dx - hleft_dx - hright_dx,
                dy,
            )
            gc.draw_image(hleft, (s_x, y, hleft_dx, dy))
            gc.draw_image(hright, (s_x + s_dx - hright_dx, y, hright_dx, dy))
            gc.draw_image(
                hmid,
                (
                    round(
                        s_x
                        + hleft_dx
                        + (s_dx - hleft_dx - hright_dx - hmid_dx) / 2.0
                    ),
                    y,
                    hmid_dx,
                    dy,
                ),
            )
            self._info = (t_x, s_x, s_x + s_dx, r_x)

    def _get_zone(self, event):
        "Determine which scrollbar zone the mouse pointer is over"
        if not self.xy_in_bounds(event) or (self._info is None):
            return NO_SCROLL
        cbl, csl, csh, ctr = self._info
        c = [event.x, event.y][self.style[0] == "v"]
        if c < cbl:
            return LINE_DOWN
        if c >= ctr:
            return LINE_UP
        if c < csl:
            return PAGE_DOWN
        if c >= csh:
            return PAGE_UP
        return SLIDER

    def _scroll(self):
        "Perform an incremental scroll (line up/down, page up/down)"
        incr = self._scroll_incr
        if incr != 0.0:
            low, high, page_size, line_size = self.range
            position = max(min(self.position + incr, high - page_size), low)
            if position == self.position:
                return
            self.position = position
        setattr(self, self._event_name, True)

    def _set_zone_suffix(self, zone, suffix):
        "Set a particular zone's image suffix"
        if zone != NO_SCROLL:
            suffix_name = zone_suffixes[zone]
            if suffix_name != "":
                setattr(self, suffix_name, suffix)
                self.redraw()

    # -------------------------------------------------------------------------
    #  Handle mouse events:
    # -------------------------------------------------------------------------

    def _left_down_changed(self, event):
        event.handled = True
        if self.range[2] == 0.0:
            return
        zone = self._get_zone(event)
        if zone != NO_SCROLL:
            self.window.mouse_owner = self
            self._scrolling = zone
            self._set_zone_suffix(zone, "_down")
            if zone == SLIDER:
                self._xy = (event.x, event.y)[self.style[0] == "v"]
                self._position = self.position
            else:
                self._event_name, sign, index = scroll_info[zone]
                line_size = self.range[3]
                incr = 0.0
                if line_size != 0.0:
                    incr = sign * self.range[index]
                    if index == 2:
                        incr -= sign * line_size
                self._scroll_incr = incr
                self._scroll()
                self._in_zone = True
                self.timer_interval = 0.5

    def _left_dclick_changed(self, event):
        self._left_down_changed(event)

    def _left_up_changed(self, event):
        event.handled = True
        scrolling = self._scrolling
        if scrolling != NO_SCROLL:
            zone = self._get_zone(event)
            self._set_zone_suffix(scrolling, "")
            self._set_zone_suffix(zone, "_over")
            if scrolling != SLIDER:
                self.timer_interval = None
            self._scrolling = NO_SCROLL
            self._zone = zone
            if zone == NO_SCROLL:
                self.window.mouse_owner = None
            self.scroll_done = True

    def _mouse_move_changed(self, event):
        event.handled = True
        self.pointer = "arrow"
        zone = self._get_zone(event)
        scrolling = self._scrolling
        if scrolling == SLIDER:
            xy = (event.x, event.y)[self.style[0] == "v"]
            low, high, page_size, line_size = self.range
            position = self._position + (
                (xy - self._xy) * (high - low - page_size) / self._range
            )
            self.position = max(min(position, high - page_size), low)
        elif scrolling != NO_SCROLL:
            in_zone = zone == scrolling
            if in_zone != self._in_zone:
                self._in_zone = in_zone
                self._set_zone_suffix(scrolling, ["", "_down"][in_zone])
        elif zone != self._zone:
            self._set_zone_suffix(self._zone, "")
            self._set_zone_suffix(zone, "_over")
            self._zone = zone
            self.window.mouse_owner = [self, None][zone == NO_SCROLL]

    def _mouse_wheel_changed(self, event):
        "Scrolls when the mouse scroll wheel is spun"
        event.handled = True
        self.position += (event.mouse_wheel * self.page_size) / 20

    def _timer_changed(self):
        "Handle timer events"
        if self._scrolling != NO_SCROLL:
            self.timer_interval = 0.1
            if self._in_zone:
                self._scroll()
