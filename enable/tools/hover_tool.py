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
Tool to detect when the user hovers over a specific part of an underlying
components.
"""

# Enthought library imports
from enable.base_tool import BaseTool
from traits.api import Any, Callable, Enum, Float, Int
from traits.etsconfig.api import ETSConfig
from pyface.timer.api import DoLaterTimer


# Define a toolkit-specific function for determining the global mouse position
if ETSConfig.toolkit == "wx":
    import wx

    def GetGlobalMousePosition():
        pos = wx.GetMousePosition()
        if isinstance(pos, tuple):
            return pos
        elif hasattr(pos, "x") and hasattr(pos, "y"):
            return (pos.x, pos.y)
        else:
            raise RuntimeError("Unable to determine mouse position")


elif ETSConfig.toolkit.startswith("qt"):
    from pyface.qt import QtGui

    def GetGlobalMousePosition():
        pos = QtGui.QCursor.pos()
        return (pos.x(), pos.y())


else:

    def GetGlobalMousePosition():
        raise NotImplementedError(
            "GetGlobalMousePosition is not defined for"
            "toolkit '%s'." % ETSConfig.toolkit
        )


class HoverTool(BaseTool):
    """
    Tool to detect when the user hovers over a certain area on a component.
    The type of area to detect can be configured by the 'area_type' and
    'bounds' traits.

    Users of the class should either set the 'callback' attribute, or
    subclass and override on_hover().
    """

    # Defines the part of the component that the hover tool will listen
    area_type = Enum(
        "top", "bottom", "left", "right", "borders",  # borders
        "UL", "UR", "LL", "LR", "corners",  # corners
    )

    # The width/height of the border or corner area.  (Corners are assumed to
    # be square.)
    area = Float(35.0)

    # The number of milliseconds that the user has to keep the mouse within
    # the hover threshold before a hover is triggered.
    hover_delay = Int(500)

    # Controls the mouse sensitivity; if the mouse moves less than the
    # threshold amount in X or Y, then the hover timer is not reset.
    hover_threshold = Int(5)

    # The action to perform when the hover activates.  This can be used
    # instead of subclassing and overriding on_hover().
    # If cb_param is not None, then the callback gets passed it as
    # its single parameter.
    callback = Callable

    # An optional parameter that gets passed to the callback.
    cb_param = Any

    # ------------------------------------------------------------------------
    # Private traits
    # ------------------------------------------------------------------------

    # A tuple (x,y) of the mouse position (in global coordinate) when we set
    # the timer
    _start_xy = Any

    # The timer
    _timer = Any

    # ------------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------------

    def on_hover(self, *args, **kwargs):
        """ This gets called when all the conditions of the hover action have
        been met, and the tool determines that the mouse is, in fact, hovering
        over a target region on the component.

        By default, this method call self.callback (if one is configured).
        """
        if self.callback is not None:
            if self.cb_param is not None:
                self.callback(self.cb_param)
            else:
                self.callback()

    def normal_mouse_move(self, event):
        if self._is_in(event.x, event.y):
            # update xy and restart the timer
            self._start_xy = GetGlobalMousePosition()
            self.restart_hover_timer(event)

    def restart_hover_timer(self, event):
        if self._timer is None:
            self._create_timer(event)
        self._timer.start()

    def on_timer(self, *args, **kwargs):
        position = GetGlobalMousePosition()
        diffx = abs(position[0] - self._start_xy[0])
        diffy = abs(position[1] - self._start_xy[1])

        if (diffx < self.hover_threshold) and (diffy < self.hover_threshold):
            self.on_hover()

    # ------------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------------

    def _is_in(self, x, y):
        """ Returns True if local coordinates (x,y) is inside our hover region
        """
        area_type = self.area_type.lower()
        c = self.component

        t = c.y2 - y <= self.area
        b = y - c.y <= self.area
        r = c.x2 - x <= self.area
        l = x - c.x <= self.area
        corner_mapping = {"ul": t & l, "ur": t & r, "ll": b & l, "lr": b & r}

        if area_type in ("top", "bottom", "left", "right"):
            return locals()[area_type[0]]
        elif area_type in ("ul", "ur", "ll", "lr"):
            return corner_mapping[area_type]
        elif area_type == "corners":
            return (t | b) & (l | r)
        elif area_type == "borders":
            return any((t, b, r, l))

    def _create_timer(self, event):
        self._timer = DoLaterTimer(self.hover_delay, self.on_timer, (), {})
