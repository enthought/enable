#------------------------------------------------------------------------------
# Copyright (c) 2008, Riverbank Computing Limited
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license.
#
# Author: Riverbank Computing Limited
# Description: <Enthought enable package component>
#
# In an e-mail to enthought-dev on 2008.09.12 at 2:49 AM CDT, Phil Thompson said:
# The advantage is that all of the PyQt code in ETS can now be re-licensed to
# use the BSD - and I hereby give my permission for that to be done. It's
# been on my list of things to do.
#------------------------------------------------------------------------------

# Qt imports.
from traits.qt import QtCore, QtGui, QtOpenGL

# Enthought library imports.
from enable.abstract_window import AbstractWindow
from enable.events import KeyEvent, MouseEvent
from traits.api import Instance

# Local imports.
from constants import BUTTON_NAME_MAP, KEY_MAP, POINTER_MAP

class _QtWindowHandler(object):
    def __init__(self, qt_window, enable_window):
        self._enable_window = enable_window

        pos = qt_window.mapFromGlobal(QtGui.QCursor.pos())
        self.last_mouse_pos = (pos.x(), pos.y())

        qt_window.setAutoFillBackground(True)
        qt_window.setFocusPolicy(QtCore.Qt.WheelFocus)
        qt_window.setMouseTracking(True)
        qt_window.setSizePolicy(QtGui.QSizePolicy.Expanding,
                                QtGui.QSizePolicy.Expanding)

    def closeEvent(self, event):
        self._enable_window.cleanup()
        self._enable_window = None

    def paintEvent(self, event):
        self._enable_window._paint(event)

    def resizeEvent(self, event):
        dx = event.size().width()
        dy = event.size().height()
        component = self._enable_window.component

        self._enable_window.resized = (dx, dy)

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

    def keyReleaseEvent(self, event):
        if self._enable_window:
            self._enable_window._handle_key_event(event)

    #------------------------------------------------------------------------
    # Qt Mouse event handlers
    #------------------------------------------------------------------------

    def enterEvent(self, event):
        if self._enable_window:
            self._enable_window._handle_mouse_event("mouse_enter", event)

    def leaveEvent(self, event):
        if self._enable_window:
            self._enable_window._handle_mouse_event("mouse_leave", event)

    def mouseDoubleClickEvent(self, event):
        if self._enable_window:
            name = BUTTON_NAME_MAP[event.button()]
            self._enable_window._handle_mouse_event(name + "_dclick", event)

    def mouseMoveEvent(self, event):
        if self._enable_window:
            self._enable_window._handle_mouse_event("mouse_move", event)

    def mousePressEvent(self, event):
        if self._enable_window:
            name = BUTTON_NAME_MAP[event.button()]
            self._enable_window._handle_mouse_event(name + "_down", event)

    def mouseReleaseEvent(self, event):
        if self._enable_window:
            name = BUTTON_NAME_MAP[event.button()]
            self._enable_window._handle_mouse_event(name + "_up", event)

    def wheelEvent(self, event):
        if self._enable_window:
            self._enable_window._handle_mouse_event("mouse_wheel", event)

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


class _QtWindow(QtGui.QWidget):
    """ The Qt widget that implements the enable control. """
    def __init__(self, enable_window):
        super(_QtWindow, self).__init__()
        self.handler = _QtWindowHandler(self, enable_window)

    def closeEvent(self, event):
        self.handler.closeEvent(event)
        return super(_QtWindow, self).closeEvent(event)

    def paintEvent(self, event):
        self.handler.paintEvent(event)

    def resizeEvent(self, event):
        self.handler.resizeEvent(event)

    def keyReleaseEvent(self, event):
        self.handler.keyReleaseEvent(event)

    def enterEvent(self, event):
        self.handler.enterEvent(event)

    def leaveEvent(self, event):
        self.handler.leaveEvent(event)

    def mouseDoubleClickEvent(self, event):
        self.handler.mouseDoubleClickEvent(event)

    def mouseMoveEvent(self, event):
        self.handler.mouseMoveEvent(event)

    def mousePressEvent(self, event):
        self.handler.mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        self.handler.mouseReleaseEvent(event)

    def wheelEvent(self, event):
        self.handler.wheelEvent(event)

    def dragEnterEvent(self, event):
        self.handler.dragEnterEvent(event)

    def dragLeaveEvent(self, event):
        self.handler.dragLeaveEvent(event)

    def dragMoveEvent(self, event):
        self.handler.dragMoveEvent(event)

    def dropEvent(self, event):
        self.handler.dropEvent(event)


class _QtGLWindow(QtOpenGL.QGLWidget):
    def __init__(self, enable_window):
        super(_QtGLWindow, self).__init__()
        self.handler = _QtWindowHandler(self, enable_window)

    def closeEvent(self, event):
        self.handler.closeEvent(event)
        return super(_QtGLWindow, self).closeEvent(event)

    def paintEvent(self, event):
        super(_QtGLWindow, self).paintEvent(event)
        self.handler.paintEvent(event)

    def resizeEvent(self, event):
        super(_QtGLWindow, self).resizeEvent(event)
        self.handler.resizeEvent(event)

    def keyReleaseEvent(self, event):
        self.handler.keyReleaseEvent(event)

    def enterEvent(self, event):
        self.handler.enterEvent(event)

    def leaveEvent(self, event):
        self.handler.leaveEvent(event)

    def mouseDoubleClickEvent(self, event):
        self.handler.mouseDoubleClickEvent(event)

    def mouseMoveEvent(self, event):
        self.handler.mouseMoveEvent(event)

    def mousePressEvent(self, event):
        self.handler.mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        self.handler.mouseReleaseEvent(event)

    def wheelEvent(self, event):
        self.handler.wheelEvent(event)

    def dragEnterEvent(self, event):
        self.handler.dragEnterEvent(event)

    def dragLeaveEvent(self, event):
        self.handler.dragLeaveEvent(event)

    def dragMoveEvent(self, event):
        self.handler.dragMoveEvent(event)

    def dropEvent(self, event):
        self.handler.dropEvent(event)


class _Window(AbstractWindow):

    control = Instance(QtGui.QWidget)

    def __init__(self, parent, wid=-1, pos=None, size=None, **traits):
        AbstractWindow.__init__(self, **traits)

        self._mouse_captured = False

        self.control = self._create_control(self)

        if pos is not None:
            self.control.move(*pos)

        if size is not None:
            self.control.resize(*size)

    #------------------------------------------------------------------------
    # Implementations of abstract methods in AbstractWindow
    #------------------------------------------------------------------------

    def set_drag_result(self, result):
        # FIXME
        raise NotImplementedError

    def _capture_mouse ( self ):
        "Capture all future mouse events"
        # Nothing needed with Qt.
        pass

    def _release_mouse ( self ):
        "Release the mouse capture"
        # Nothing needed with Qt.
        pass
    
    def _create_key_event(self, event):
        focus_owner = self.focus_owner

        if focus_owner is None:
            focus_owner = self.component

            if focus_owner is None:
                event.ignore()
                return None

        # Convert the keypress to a standard enable key if possible, otherwise
        # to text.
        key = KEY_MAP.get(event.key())

        if key is None:
            key = unicode(event.text())

            if not key:
                return None

        # Use the last-seen mouse position as the coordinates of this event.
        x, y = self.control.last_mouse_pos

        modifiers = event.modifiers()

        return KeyEvent(character=key, x=x,
                        y=self._flip_y(y),
                        alt_down=bool(modifiers & QtCore.Qt.AltModifier),
                        shift_down=bool(modifiers & QtCore.Qt.ShiftModifier),
                        control_down=bool(modifiers & QtCore.Qt.ControlModifier),
                        event=event,
                        window=self)

    def _create_mouse_event(self, event):
        # If the event (if there is one) doesn't contain the mouse position,
        # modifiers and buttons then get sensible defaults.
        try:
            x = event.x()
            y = event.y()
            modifiers = event.modifiers()
            buttons = event.buttons()
        except AttributeError:
            pos = self.control.mapFromGlobal(QtGui.QCursor.pos())
            x = pos.x()
            y = pos.y()
            modifiers = 0
            buttons = 0

        self.control.last_mouse_pos = (x, y)

        # A bit crap, because AbstractWindow was written with wx in mind, and
        # we treat wheel events like mouse events.
        if isinstance(event, QtGui.QWheelEvent):
            delta = event.delta()
            degrees_per_step = 15.0
            mouse_wheel = delta / float(8 * degrees_per_step)
        else:
            mouse_wheel = 0

        return MouseEvent(x=x, y=self._flip_y(y), mouse_wheel=mouse_wheel,
                alt_down=bool(modifiers & QtCore.Qt.AltModifier),
                shift_down=bool(modifiers & QtCore.Qt.ShiftModifier),
                control_down=bool(modifiers & QtCore.Qt.ControlModifier),
                left_down=bool(buttons & QtCore.Qt.LeftButton),
                middle_down=bool(buttons & QtCore.Qt.MidButton),
                right_down=bool(buttons & QtCore.Qt.RightButton),
                window=self)

    def _redraw(self, coordinates=None):
        if self.control:
            if coordinates is None:
                self.control.update()
            else:
                self.control.update(*coordinates)

    def _get_control_size(self):
        if self.control:
            return (self.control.width(), self.control.height())

        return None

    def _create_gc(self, size, pix_format="bgra32"):
        raise NotImplementedError

    def _window_paint(self, event):
        raise NotImplementedError

    def set_pointer(self, pointer):
        self.control.setCursor(POINTER_MAP[pointer])

    def _set_timer_interval(self, component, interval):
        # FIXME
        raise NotImplementedError

    def set_tooltip(self, tooltip):
        self.control.setToolTip(tooltip)

    def _set_focus(self):
        self.control.setFocus()

    #------------------------------------------------------------------------
    # Private methods
    #------------------------------------------------------------------------

    def _flip_y(self, y):
        "Converts between a Kiva and a Qt y coordinate"
        return int(self._size[1] - y - 1)


class BaseGLWindow(_Window):
    # The toolkit control
    control = Instance(_QtGLWindow)

    def _create_control(self, enable_window):
        """ Create the toolkit control.
        """
        return _QtGLWindow(enable_window)


class BaseWindow(_Window):
    # The toolkit control
    control = Instance(_QtWindow)

    def _create_control(self, enable_window):
        """ Create the toolkit control.
        """
        return _QtWindow(enable_window)
