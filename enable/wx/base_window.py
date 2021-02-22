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
Defines the concrete top-level Enable 'Window' class for the wxPython GUI
toolkit, based on the kiva agg driver.
"""

import sys
import time
import warnings

import wx

from traits.api import Any, Instance
from traitsui.wx.menu import MakeMenu

# Relative imports
from enable.events import MouseEvent, KeyEvent, DragEvent
from enable.abstract_window import AbstractWindow

from .constants import (
    DRAG_RESULTS_MAP, KEY_MAP, MOUSE_WHEEL_AXIS_MAP, POINTER_MAP
)

try:
    from pyface.wx.drag_and_drop import clipboard, PythonDropTarget
except ImportError:
    warnings.warn("pyface.wx is unavailable", ImportWarning)
    clipboard = None
    PythonDropTarget = None

# -----------------------------------------------------------------------------
#  Constants:
# -----------------------------------------------------------------------------

# Number of pixels to scroll at a time:
scroll_incr = 16

# Reusable instance to avoid constructor/destructor overhead:
wx_rect = wx.Rect(0, 0, 0, 0)

# Default 'fake' start event for wxPython based drag operations:
default_start_event = MouseEvent()


# To conserve system resources, there is only one 'timer' per program:
system_timer = None


class EnableTimer(wx.Timer):
    """ This class maintains a 'work list' of scheduled components, where
    each item in the list has the form: [ component, interval, timer_pop_time ]
    """

    def __init__(self):
        wx.Timer.__init__(self)
        self._work_list = []

    def schedule(self, component, interval):
        """ Schedule a timer event for a specified component
        """
        work_list = self._work_list
        if len(work_list) == 0:
            self.Start(5, oneShot=False)
        for i, item in enumerate(work_list):
            if component is item[0]:
                del work_list[i]
                break
        self.reschedule(component, interval)

    def reschedule(self, component, interval):
        """ Reshedule a recurring timer event for a component
        """
        pop_time = time.time() + interval
        new_item = [component, interval, pop_time]
        work_list = self._work_list
        for i, item in enumerate(work_list):
            if pop_time < item[2]:
                work_list.insert(i, new_item)
                break
        else:
            work_list.append(new_item)

    def cancel(self, component):
        """ Cancel any pending timer events for a component
        """
        work_list = self._work_list
        for i, item in enumerate(work_list):
            if component is item[0]:
                del work_list[i]
                if len(work_list) == 0:
                    self.Stop()
                break
        return len(work_list) != 0

    def Notify(self):
        """ Handle a timer 'pop' event; used for performance testing purposes
        """
        now = time.time()
        work_list = self._work_list
        n = len(work_list)
        i = 0
        while (i < n) and (now >= work_list[i][2]):
            i += 1
        if i > 0:
            reschedule = work_list[:i]
            del work_list[:i]
            for component, interval, ignore in reschedule:
                self.reschedule(component, interval)
                component.timer = True


class LessSuckyDropTarget(PythonDropTarget):
    """ The sole purpose of this class is to override the implementation
    of OnDragOver() in the parent class to NOT short-circuit return the
    'default_drag_result' if the drop_source is None.  (The parent class
    implementation basically means that everything looks like it's OK to
    drop, and the DnD handler doesn't ever get a chance to intercept or
    veto.)
    """

    def OnDragOver(self, x, y, default_drag_result):
        # This is a cut-and-paste job of the parent class implementation.
        # Please refer to its comments.

        if (clipboard.drop_source is not None
                and not clipboard.drop_source.allow_move):
            default_drag_result = wx.DragCopy

        if hasattr(self.handler, "wx_drag_over"):
            drag_result = self.handler.wx_drag_over(
                x, y, clipboard.data, default_drag_result
            )
        else:
            drag_result = default_drag_result

        return drag_result


class BaseWindow(AbstractWindow):

    # Screen scroll increment amount:
    scroll_incr = (wx.SystemSettings.GetMetric(wx.SYS_SCREEN_Y) or 768) / 20

    # Width/Height of standard scrollbars:
    scrollbar_dx = wx.SystemSettings.GetMetric(wx.SYS_VSCROLL_X)
    scrollbar_dy = wx.SystemSettings.GetMetric(wx.SYS_HSCROLL_Y)

    _cursor_color = Any  # PZW: figure out the correct type for this...

    # Reference to the actual wxPython window:
    control = Instance(wx.Window)

    # This is set by downstream components to notify us of whether or not
    # the current drag operation should return DragCopy, DragMove, or DragNone.
    _drag_result = Any

    def __init__(self, parent, wid=-1, pos=wx.DefaultPosition,
                 size=wx.DefaultSize, **traits):
        AbstractWindow.__init__(self, **traits)
        self._timer = None
        self._mouse_captured = False

        # Due to wx wonkiness, we don't reliably get cursor position from
        # a wx KeyEvent.  Thus, we manually keep track of when we last saw
        # the mouse and use that information instead.  These coordinates are
        # in the wx coordinate space, i.e. pre-self._flip_y().
        self._last_mouse_pos = (0, 0)

        # Create the delegate:
        self.control = control = self._create_control(parent, wid, pos, size)

        # Figure out the pixel scale factor
        if self.high_resolution:
            self.base_pixel_scale = self.control.GetContentScaleFactor()

        # Set up the 'erase background' event handler:
        control.Bind(wx.EVT_ERASE_BACKGROUND, self._on_erase_background)

        # Set up the 'paint' event handler:
        control.Bind(wx.EVT_PAINT, self._paint)
        control.Bind(wx.EVT_SIZE, self._on_size)

        # Set up mouse event handlers:
        control.Bind(wx.EVT_LEFT_DOWN, self._on_left_down)
        control.Bind(wx.EVT_LEFT_UP, self._on_left_up)
        control.Bind(wx.EVT_LEFT_DCLICK, self._on_left_dclick)
        control.Bind(wx.EVT_MIDDLE_DOWN, self._on_middle_down)
        control.Bind(wx.EVT_MIDDLE_UP, self._on_middle_up)
        control.Bind(wx.EVT_MIDDLE_DCLICK, self._on_middle_dclick)
        control.Bind(wx.EVT_RIGHT_DOWN, self._on_right_down)
        control.Bind(wx.EVT_RIGHT_UP, self._on_right_up)
        control.Bind(wx.EVT_RIGHT_DCLICK, self._on_right_dclick)
        control.Bind(wx.EVT_MOTION, self._on_mouse_move)
        control.Bind(wx.EVT_ENTER_WINDOW, self._on_window_enter)
        control.Bind(wx.EVT_LEAVE_WINDOW, self._on_window_leave)
        control.Bind(wx.EVT_MOUSEWHEEL, self._on_mouse_wheel)

        # Handle key up/down events:
        control.Bind(wx.EVT_KEY_DOWN, self._on_key_pressed)
        control.Bind(wx.EVT_KEY_UP, self._on_key_released)
        control.Bind(wx.EVT_CHAR, self._on_character)

        # Attempt to allow wxPython drag and drop events to be mapped to
        # Enable drag events:

        # Handle window close and cleanup
        control.Bind(wx.EVT_WINDOW_DESTROY, self._on_close)

        if PythonDropTarget is not None:
            control.SetDropTarget(LessSuckyDropTarget(self))
            self._drag_over = []

        # In some cases, on the Mac at least, we never get an initial EVT_SIZE
        # since the initial size is correct. Because of this we call _on_size
        # here to initialize our bounds.
        self._on_size(None)

    def _create_control(self, parent, wid, pos=wx.DefaultPosition,
                        size=wx.DefaultSize):
        if parent is None:
            # pab hack in order to avoid that the wx.Window crash!
            parent = wx.Frame(None)
        return wx.Window(
            parent, wid, pos, size, style=wx.CLIP_CHILDREN | wx.WANTS_CHARS
        )

    def _on_close(self, event):
        # Might be scrollbars or other native components under
        # us that are generating this event
        if event.GetWindow() == self.control:
            self._gc = None
            self.control.Bind(wx.EVT_ERASE_BACKGROUND, None)
            self.control.Bind(wx.EVT_PAINT, None)
            self.control.Bind(wx.EVT_SIZE, None)
            self.control.Bind(wx.EVT_LEFT_DOWN, None)
            self.control.Bind(wx.EVT_LEFT_UP, None)
            self.control.Bind(wx.EVT_LEFT_DCLICK, None)
            self.control.Bind(wx.EVT_MIDDLE_DOWN, None)
            self.control.Bind(wx.EVT_MIDDLE_UP, None)
            self.control.Bind(wx.EVT_MIDDLE_DCLICK, None)
            self.control.Bind(wx.EVT_RIGHT_DOWN, None)
            self.control.Bind(wx.EVT_RIGHT_UP, None)
            self.control.Bind(wx.EVT_RIGHT_DCLICK, None)
            self.control.Bind(wx.EVT_MOTION, None)
            self.control.Bind(wx.EVT_ENTER_WINDOW, None)
            self.control.Bind(wx.EVT_LEAVE_WINDOW, None)
            self.control.Bind(wx.EVT_MOUSEWHEEL, None)
            self.control.Bind(wx.EVT_KEY_DOWN, None)
            self.control.Bind(wx.EVT_KEY_UP, None)
            self.control.Bind(wx.EVT_CHAR, None)
            self.control.Bind(wx.EVT_WINDOW_DESTROY, None)
            self.control.SetDropTarget(None)
            self.control = None
            self.component.cleanup(self)
            self.component.parent = None
            self.component.window = None
            self.component = None

    def _flip_y(self, y):
        """ Convert from a Kiva to a wxPython y coordinate
        """
        # Handle the device pixel ratio adjustment here
        return int(self._size[1] / self.base_pixel_scale - 1 - y)

    def _on_erase_background(self, event):
        pass

    def _on_size(self, event):
        dx, dy = self.control.GetSize()

        # do nothing if the new and old sizes are the same
        if ((self.component.outer_width, self.component.outer_height)
                == (dx, dy)):
            return

        self.resized = (dx, dy)

        if getattr(self.component, "fit_window", False):
            self.component.outer_position = [0, 0]
            self.component.outer_bounds = [dx, dy]
        elif hasattr(self.component, "resizable"):
            if "h" in self.component.resizable:
                self.component.outer_x = 0
                self.component.outer_width = dx
            if "v" in self.component.resizable:
                self.component.outer_y = 0
                self.component.outer_height = dy

        self.control.Refresh()

    def _capture_mouse(self):
        """ Capture all future mouse events
        """
        if not self._mouse_captured:
            self._mouse_captured = True
            self.control.CaptureMouse()

    def _release_mouse(self):
        """ Release the mouse capture
        """
        if self._mouse_captured:
            self._mouse_captured = False
            self.control.ReleaseMouse()

    def _on_key_pressed(self, event):
        handled = self._handle_key_event("key_pressed", event)
        if not handled:
            event.Skip()

    def _on_key_released(self, event):
        handled = self._handle_key_event("key_released", event)
        if not handled:
            event.Skip()

    def _create_key_event(self, event_type, event):
        """ Convert a GUI toolkit keyboard event into a KeyEvent.
        """
        if self.focus_owner is None:
            focus_owner = self.component
        else:
            focus_owner = self.focus_owner

        if focus_owner is not None:
            if event_type == "character":
                key = chr(event.GetUnicodeKey())
                if not key:
                    return None
            else:
                key_code = event.GetKeyCode()
                if key_code in KEY_MAP:
                    key = KEY_MAP.get(key_code)
                else:
                    key = chr(event.GetUnicodeKey()).lower()

            # Use the last-seen mouse coordinates instead of GetX/GetY due
            # to wx bug.
            x, y = self._last_mouse_pos

            # Someday when wx does this properly, we can use these instead:
            # x = event.GetX()
            # y = event.GetY()

            return KeyEvent(
                event_type=event_type,
                character=key,
                alt_down=event.AltDown(),
                control_down=event.ControlDown(),
                shift_down=event.ShiftDown(),
                x=x,
                y=self._flip_y(y),
                event=event,
                window=self,
            )
        else:
            event.Skip()

        return None

    def _create_mouse_event(self, event):
        """ Convert a GUI toolkit mouse event into a MouseEvent
        """
        if event is not None:
            x = event.GetX()
            y = event.GetY()
            self._last_mouse_pos = (x, y)

            mouse_wheel = (
                float(event.GetLinesPerAction()) * event.GetWheelRotation()
            ) / (event.GetWheelDelta() or 1)
            wheel_axis = MOUSE_WHEEL_AXIS_MAP[event.GetWheelAxis()]
            if wheel_axis == "horizontal":
                mouse_wheel_delta = (mouse_wheel * 120 / 200.0, 0)
            else:
                mouse_wheel_delta = (0, mouse_wheel * 120 / 200.0)

            # Note: The following code fixes a bug in wxPython that returns
            # 'mouse_wheel' events in screen coordinates, rather than window
            # coordinates:
            if wx.VERSION[:2] < (2, 8):
                if mouse_wheel != 0 and sys.platform == "win32":
                    x, y = self.control.ScreenToClientXY(x, y)
            return MouseEvent(
                x=x,
                y=self._flip_y(y),
                alt_down=event.AltDown(),
                control_down=event.ControlDown(),
                shift_down=event.ShiftDown(),
                left_down=event.LeftIsDown(),
                middle_down=event.MiddleIsDown(),
                right_down=event.RightIsDown(),
                mouse_wheel=mouse_wheel,
                mouse_wheel_axis=wheel_axis,
                mouse_wheel_delta=mouse_wheel_delta,
                window=self,
            )

        # If no event specified, make one up:
        x, y = wx.GetMousePosition()
        x, y = self.control.ScreenToClientXY(x, y)
        self._last_mouse_pos = (x, y)
        return MouseEvent(
            x=x,
            y=self._flip_y(y),
            alt_down=self.alt_pressed,
            control_down=self.ctrl_pressed,
            shift_down=self.shift_pressed,
            left_down=False,
            middle_down=False,
            right_down=False,
            mouse_wheel=0,
            window=self,
        )

    def _create_gc(self, size, pix_format=None):
        """ Create a Kiva graphics context of a specified size
        """
        raise NotImplementedError

    def _redraw(self, coordinates=None):
        """ Request a redraw of the window """
        if coordinates is None:
            if self.control:
                self.control.Refresh(False)
        else:
            xl, yb, xr, yt = coordinates
            rect = wx_rect
            rect.SetX(int(xl))
            rect.SetY(int(self._flip_y(yt - 1)))
            rect.SetWidth(int(xr - xl))
            rect.SetHeight(int(yt - yb))
            if self.control:
                self.control.Refresh(False, rect)

    def _get_control_size(self):
        """ Get the size of the underlying toolkit control
        """
        result = None
        if self.control:
            result = self.control.GetSize() * self.base_pixel_scale
        return result

    def _window_paint(self, event):
        """ Do a GUI toolkit specific screen update
        """
        raise NotImplementedError

    def set_pointer(self, pointer):
        """ Set the current pointer (i.e. cursor) shape
        """
        ptr = POINTER_MAP[pointer]
        if isinstance(ptr, type(wx.CURSOR_ARROW)):
            POINTER_MAP[pointer] = ptr = wx.Cursor(ptr)
        self.control.SetCursor(ptr)

    def set_tooltip(self, tooltip):
        """ Set the current tooltip for the window
        """
        wx.ToolTip_Enable(False)
        self.control.SetToolTip(wx.ToolTip(tooltip))
        wx.ToolTip_Enable(True)

    def set_timer_interval(self, component, interval):
        """ Set up or cancel a timer for a specified component.  To cancel the
        timer, set interval=None
        """
        global system_timer
        if interval is None:
            if (system_timer is not None
                    and not system_timer.cancel(component)):
                system_timer = None
        else:
            if system_timer is None:
                system_timer = EnableTimer()
            system_timer.schedule(component, interval)

    def _set_focus(self):
        """ Sets the keyboard focus to this window
        """
        self.control.SetFocus()

    def screen_to_window(self, x, y):
        pt = wx.Point(x, y)
        x, y = self.control.ScreenToClient(pt)
        y = self._flip_y(y)
        return x, y

    def get_pointer_position(self):
        """ Returns the current pointer position in local window coordinates
        """
        pos = wx.GetMousePosition()
        return self.screen_to_window(pos.x, pos.y)

    def set_drag_result(self, result):
        if result not in DRAG_RESULTS_MAP:
            raise RuntimeError("Unknown drag result '%s'" % result)
        self._drag_result = DRAG_RESULTS_MAP[result]

    def wx_dropped_on(self, x, y, drag_object, drop_result):
        """ Handle wxPython drag and drop events
        """
        # Process the 'dropped_on' event for the object(s) it was dropped on:
        y = self._flip_y(y)
        drag_event = DragEvent(x=x, y=y, obj=drag_object, window=self)
        self._drag_result = wx.DragNone
        if self.component.is_in(x, y):
            self.component.dispatch(drag_event, "dropped_on")

        # If a downstream component wants to express that it handled the
        return self._drag_result

    def wx_drag_over(self, x, y, drag_object, drag_result):
        y = self._flip_y(y)
        drag_over_event = DragEvent(
            x=x,
            y=y,
            x0=0.0,
            y0=0.0,
            copy=drag_result != wx.DragMove,
            obj=drag_object,
            start_event=default_start_event,
            window=self,
        )

        # By default, don't indicate that we can be dropped on.  It is up
        # to the component to set this correctly.
        self._drag_result = wx.DragNone
        if self.component.is_in(x, y):
            self.component.dispatch(drag_over_event, "drag_over")

        return self._drag_result

    def wx_drag_leave(self, drag_object):
        drag_leave_event = DragEvent(
            x=0.0,
            y=0.0,
            x0=0.0,
            y0=0.0,
            copy=False,
            obj=drag_object,
            start_event=default_start_event,
            window=self,
        )
        self.component.dispatch(drag_leave_event, "drag_leave")

    def create_menu(self, menu_definition, owner):
        """ Create a wxMenu from a string description
        """
        return MakeMenu(menu_definition, owner, True, self.control)

    def popup_menu(self, menu, x, y):
        """ Pop-up a wxMenu at a specified location
        """
        self.control.PopupMenuXY(menu.menu, int(x), int(self._flip_y(y)))
