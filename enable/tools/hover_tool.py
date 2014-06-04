"""
Tool to detect when the user hovers over a specific part of an underlying
components.
"""

from __future__ import absolute_import

# Enthought library imports
from enable.base_tool import BaseTool
from traits.etsconfig.api import ETSConfig
from pyface.toolkit import toolkit_object
from traits.api import Any, Callable, Enum, Float, Int

# Define a toolkit-specific function for determining the global mouse position
if ETSConfig.toolkit == 'wx':
    import wx
    def GetGlobalMousePosition():
        pos = wx.GetMousePosition()
        if isinstance(pos, tuple):
            return pos
        elif hasattr(pos, "x") and hasattr(pos, "y"):
            return (pos.x, pos.y)
        else:
            raise RuntimeError("Unable to determine mouse position")

elif ETSConfig.toolkit == 'qt4':
    from pyface.qt import QtGui
    def GetGlobalMousePosition():
        pos = QtGui.QCursor.pos()
        return (pos.x(), pos.y())

else:
    def GetGlobalMousePosition():
        raise NotImplementedError, "GetGlobalMousePosition is not defined for" \
            "toolkit '%s'." % ETSConfig.toolkit


class HoverTool(BaseTool):
    """
    Tool to detect when the user hovers over a certain area on a component.
    The type of area to detect can be configured by the 'area_type' and 'bounds'
    traits.

    Users of the class should either set the 'callback' attribute, or
    subclass and override on_hover().
    """

    # Defines the part of the component that the hover tool will listen
    area_type = Enum("top", "bottom", "left", "right", "borders",   # borders
                     "UL", "UR", "LL", "LR", "corners")             # corners

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

    #-------------------------------------------------------------------------
    # Private traits
    #-------------------------------------------------------------------------

    # A tuple (x,y) of the mouse position (in global coordinate) when we set the timer
    _start_xy = Any

    # The timer
    _timer = Any


    #-------------------------------------------------------------------------
    # Public methods
    #-------------------------------------------------------------------------

    def on_hover(self):
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
        else:
            if self._timer:
                self._timer.Stop()

    def restart_hover_timer(self, event):
        if self._timer is None:
            self._create_timer(event)
        else:
            self._timer.Start()

    def on_timer(self):
        position = GetGlobalMousePosition()
        diffx = abs(position[0] - self._start_xy[0])
        diffy = abs(position[1] - self._start_xy[1])

        if (diffx < self.hover_threshold) and (diffy < self.hover_threshold):
            self.on_hover()

        self._timer.Stop()


    #-------------------------------------------------------------------------
    # Private methods
    #-------------------------------------------------------------------------

    def _is_in(self, x, y):
        """ Returns True if local coordinates (x,y) is inside our hover region """
        area_type = self.area_type.lower()
        c = self.component

        t = (c.y2 - y <= self.area)
        b = (y - c.y <= self.area)
        r = (c.x2 - x <= self.area)
        l = (x - c.x <= self.area)

        if area_type in ("top", "bottom", "left", "right"):
            return locals()[area_type[0]]
        elif area_type.lower() in ("ul", "ur", "ll", "lr"):
            u = t
            return locals()[area_type[0]] and locals()[area_type[1]]
        elif area_type == "corners":
            return (t | b) & (l | r)
        elif area_type == "borders":
            return any((t, b, r, l))

    def _create_timer(self, event):
        klass = toolkit_object("timer.timer:Timer")
        self._timer = klass(self.hover_delay, self.on_timer)
