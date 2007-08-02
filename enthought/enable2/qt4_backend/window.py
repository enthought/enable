#------------------------------------------------------------------------------
# Copyright (c) 2007, Riverbank Computing Limited
# All rights reserved.
#
# This software is provided without warranty under the terms of the GPL v2
# license.
#
# Author: Riverbank Computing Limited
# Description: <Enthought enable2 package component>
#------------------------------------------------------------------------------


# Qt imports.
from PyQt4 import QtCore, QtGui

# Enthought library imports.
from enthought.enable2.abstract_window import AbstractWindow
from enthought.enable2.events import KeyEvent, MouseEvent
from enthought.enable2.graphics_context import GraphicsContextEnable
from enthought.traits.api import Instance

# Local imports.
from constants import BUTTON_NAME_MAP, KEY_MAP, POINTER_MAP


class _QtWindow(QtGui.QWidget):
    """ The Qt widget that implements the enable control. """

    def __init__(self, enable_window, parent):
        QtGui.QWidget.__init__(self, parent)

        #self._enable_window = enable_window

        #pos = self.mapFromGlobal(QtGui.QCursor.pos())
        #self.last_mouse_pos = (pos.x(), pos.y())

        #self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent)
        #self.setFocusPolicy(QtCore.Qt.WheelFocus)
        #self.setMouseTracking(True)

    def _event(self, e):
        print "Got event", e, e.type()
        return False

    def paintEvent(self, event):
        print "In paintEvent()"
        #self._enable_window._paint(event)
        print "Leaving paintEvent()"

    def resizeEvent(self, event):
        dx = self.width()
        dy = self.height()
        print "In resizeEvent()", dx, dy
        return

        self._enable_window.resized = (dx, dy)

        component = self._enable_window.component

        if hasattr(component, "fit_window") and component.fit_window:
            component.outer_position = [0, 0]
            component.outer_bounds = [dx, dy]
        elif hasattr(component, "resizable"):
            if "h" in component.resizable:
                component.outer_x = 0
                component.outer_width = dx
            if "v" in component.resizable:
                component.outer_y = 0
                component.outer_height = dy
        print "Leaving resizeEvent()"

    def closeEvent(self, event):
        print "In closeEvent()"
        return

        if self._enable_window.component is not None:
            self._enable_window.component.cleanup(self)
            self._enable_window.component.parent = None
            self._enable_window.component.window = None
            self._enable_window.component = None

        if self.control is not None:
            self._enable_window.control = None


class Window(AbstractWindow):

    control = Instance(_QtWindow)

    def __init__(self, parent, wid=-1, pos=None, size=None, **traits):
        AbstractWindow.__init__(self, **traits)

        self._mouse_captured = False

        self.control = _QtWindow(self, parent)

        if pos is not None:
            self.control.move(*pos)

        if size is not None:
            self.control.resize(*size)
        print "In __init__()", size

    #------------------------------------------------------------------------
    # Qt Mouse event handlers
    #------------------------------------------------------------------------

    def enterEvent(self, event):
        self._handle_mouse_event("mouse_enter", event)

    def leaveEvent(self, event):
        self._handle_mouse_event("mouse_leave", event)

    def mouseDoubleClickEvent(self, event):
        name = BUTTON_NAME_MAP[event.button()]
        self._handle_mouse_event(name + "_dclick", event)

    def mouseMoveEvent(self, event):
        self._handle_mouse_event("mouse_move", event)

    def mousePressEvent(self, event):
        name = BUTTON_NAME_MAP[event.button()]
        self._handle_mouse_event(name + "_down", event)

    def mouseReleaseEvent(self, event):
        name = BUTTON_NAME_MAP[event.button()]
        self._handle_mouse_event(name + "_up", event)

    def wheelEvent(self, event):
        self._handle_mouse_event("mouse_wheel", event)

    #------------------------------------------------------------------------
    # Qt Drag and drop event handlers
    #------------------------------------------------------------------------

    def dragEnterEvent(self, event):
        pass

    def dragLeaveEvent(self, event):
        pass

    def dragMoveEvent(self, event):
        pass

    def dropEvent(self, event):
        pass

    #------------------------------------------------------------------------
    # Qt keyboard event handlers
    #------------------------------------------------------------------------

    def keyPressEvent(self, event):
        # Enable doesn't have a key_down event yet, so ignore
        pass

    def keyReleaseEvent(self, event):
        focus_owner = self.focus_owner
        if focus_owner is None:
            focus_owner = self.component

        key_code = event.key()
        key = KEY_MAP.get(key_code, None)
        if key is None:
            if key_code >= 0x20 and key_code < 256:
                # handle all of the standard ASCII table
                key = chr(key_code)

        modifiers = event.modifiers()

        # Use the last-seen mouse position as the coordinates of this event
        x, y =  self.last_mouse_pos

        enable_event = KeyEvent(character = key,
                            alt_down = modifiers & QtCore.Qt.AltModifier,
                            shift_down = modifiers & QtCore.Qt.ShiftModifier,
                            control_down = modifiers & QtCore.Qt.ControlModifier,
                            x = x,
                            y = self._flip_y(y),
                            event = event,
                            window = self)

        if focus_owner is not None:
            focus_owner.dispatch(enable_event, "key_pressed")
        else:
            event.ignore()

    #------------------------------------------------------------------------
    # Implementations of abstract methods in AbstractWindow
    #------------------------------------------------------------------------

    def set_drag_result(self, result):
        print "In set_drag_result()"
        raise NotImplementedError

    def _capture_mouse ( self ):
        "Capture all future mouse events"
        print "In _capture_mouse()"
        # TODO: Investigate whether this is even necessary with Qt, since it
        # grabs the mouse if a key is pressed.
        if not self._mouse_captured:
            self._mouse_captured = True
            self.control.grabMouse()
        return

    def _release_mouse ( self ):
        "Release the mouse capture"
        print "In _release_mouse()"
        # TODO: see TODO in _capture_mouse()
        if self._mouse_captured:
            self._mouse_captured = False
            self.control.releaseMouse()
        return

    def _create_mouse_event(self, event):
        print "In _create_mouse_event()"
        if event is not None:
            x = event.x()
            y = event.y()
            self.control.last_mouse_pos = (x,y)

            # A bit crap, because AbstractWindow was written with WX in mind, and
            # we treat wheel events like mouse events
            if isinstance(event, QtGui.QWheelEvent):
                delta = event.delta()
                degrees_per_step = 15.0
                mouse_wheel = delta / float(8 * degrees_per_step)
            else:
                mouse_wheel = 0

            modifiers = event.modifiers()
            buttons = event.buttons()

        else:
            # If no event was specified, make one up
            pos = self.control.mapFromGlobal(QtGui.QCursor.pos())
            x = pos.x()
            y = pos.y()
            self.control.last_mouse_pos = (x,y)
            mouse_wheel = 0
            modifiers = 0
            buttons = 0

        return MouseEvent(x=x, y=self._flip_y(y), mouse_wheel=mouse_wheel,
                        alt_down = modifiers & QtCore.Qt.AltModifier,
                        shift_down = modifiers & QtCore.Qt.ShiftModifier,
                        control_down = modifiers & QtCore.Qt.ControlModifier,
                        left_down = buttons & QtCore.Qt.LeftButton,
                        middle_down = buttons & QtCore.Qt.MidButton,
                        right_down = buttons & QtCore.Qt.RightButton,
                        window = self)

    def _redraw(self, coordinates=None):
        print "In _redraw()"
        if self.control:
            if coordinates is None:
                self.control.update()
            else:
                self.control.update(*coordinates)

    def _get_control_size(self):
        print "In _get_control_size()", self.control
        if self.control:
            print "Returning:", self.control.width(), self.control.height()
            return (self.control.width(), self.control.height())

        print "Returning None"
        return None

    #def _create_gc(self, size, pix_format="bgr24"):
    def _create_gc(self, size, pix_format="bgra32"):
        print "In _create_gc(), pix_format:", pix_format
        gc = GraphicsContextEnable((size[0]+1, size[1]+1),
                pix_format=pix_format, window=self)
        gc.translate_ctm(0.5, 0.5)

        return gc

    def _window_paint(self, event):
        #s = self._gc.pixel_map.convert_to_argb32string()
        print "In _window_paint()"
        #img = QtGui.QImage(s, self._gc.width(), self._gc.height(),
        #        QtGui.QImage.Format_ARGB32)
        #p = QtGui.QPainter(self.control)
        #p.drawPixmap(0, 0, QtGui.QPixmap.fromImage(img))
        print "Leaving _window_paint()"

    def set_pointer(self, pointer):
        print "In set_pointer()"
        ptr = POINTER_MAP[pointer]
        self.control.SetCursor(ptr)

    def _set_timer_interval(self, component, interval):
        print "In _set_timer_interval()"
        raise NotImplementedError

    def set_tooltip(self, tooltip):
        print "In set_tooltip()"
        pass

    def _set_focus(self):
        print "In _set_focus()"
        self.control.setFocus(QtCore.Qt.OtherFocusReason)

    #------------------------------------------------------------------------
    # Private methods
    #------------------------------------------------------------------------

    def _flip_y(self, y):
        "Converts between a Kiva and a Qt y coordinate"
        return int(self._size[1] - y - 1)
