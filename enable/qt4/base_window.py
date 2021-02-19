# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
# -----------------------------------------------------------------------------
# Copyright (c) 2008, Riverbank Computing Limited
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license.
#
# Author: Riverbank Computing Limited
# Description: <Enthought enable package component>
#
# In an e-mail to enthought-dev on 2008.09.12 at 2:49 AM CDT,
# Phil Thompson said:
# The advantage is that all of the PyQt code in ETS can now be re-licensed to
# use the BSD - and I hereby give my permission for that to be done. It's
# been on my list of things to do.
# -----------------------------------------------------------------------------
import warnings

# Qt imports.
from pyface.qt import QtCore, QtGui, QtOpenGL

# Enthought library imports.
from enable.abstract_window import AbstractWindow
from enable.events import KeyEvent, MouseEvent, DragEvent
from traits.api import Instance

# Local imports.
from .constants import (
    BUTTON_NAME_MAP,
    KEY_MAP,
    MOUSE_WHEEL_AXIS_MAP,
    POINTER_MAP,
    DRAG_RESULTS_MAP,
)


is_qt4 = QtCore.__version_info__[0] <= 4


class _QtWindowHandler(object):
    def __init__(self, qt_window, enable_window):
        self._enable_window = enable_window

        pos = qt_window.mapFromGlobal(QtGui.QCursor.pos())
        self.last_mouse_pos = (pos.x(), pos.y())

        self.in_paint_event = False

        qt_window.setAutoFillBackground(True)
        qt_window.setFocusPolicy(QtCore.Qt.WheelFocus)
        qt_window.setMouseTracking(True)
        qt_window.setSizePolicy(
            QtGui.QSizePolicy.Expanding, QtGui.QSizePolicy.Expanding
        )
        # We prevent context menu events being generated from inside this
        # widget. If a containing parent widget handles a context menu event,
        # then Enable might not get the right-click events. Enable does not
        # represent context menu events in its Event API. Users should use the
        # ContextMenuTool or just handle the right-click explicitly in other
        # ways.
        qt_window.setContextMenuPolicy(QtCore.Qt.PreventContextMenu)

    def closeEvent(self, event):
        self._enable_window.cleanup()
        self._enable_window = None

    def paintEvent(self, event):
        self.in_paint_event = True
        self._enable_window._paint(event)
        self.in_paint_event = False

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

    # ------------------------------------------------------------------------
    # Qt Keyboard event handlers
    # ------------------------------------------------------------------------

    def keyPressEvent(self, event):
        handled = False
        if self._enable_window:
            handled = self._enable_window._handle_key_event(
                "key_pressed", event)
            if not handled:
                # for consistency with wx, we only generate character events if
                # key_pressed not handled
                handled = self._enable_window._handle_key_event(
                    "character", event)
        if not handled:
            # Allow the parent Qt widget handle the event.
            event.ignore()

    def keyReleaseEvent(self, event):
        handled = False
        if self._enable_window:
            handled = self._enable_window._handle_key_event(
                "key_released", event)
        if not handled:
            # Allow the parent Qt widget handle the event.
            event.ignore()

    # ------------------------------------------------------------------------
    # Qt Mouse event handlers
    # ------------------------------------------------------------------------

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
        handled = False
        if self._enable_window:
            handled = self._enable_window._handle_mouse_event(
                "mouse_wheel", event)
        if not handled:
            # Allow the parent Qt widget handle the event.
            event.ignore()

    def sizeHint(self, qt_size_hint):
        """ Combine the Qt and enable size hints.

        Combine the size hint coming from the Qt component (usually -1, -1)
        with the preferred size of the enable component and the size
        of the enable window.

        The combined size hint is
        - the Qt size hint if larger than 0
        - the maximum of the plot's preferred size and the window size
          (component-wise)

        E.g., if
        qt size hint = (-1, -1)
        component preferred size = (500, 200)
        size of enable window = (400, 400)

        the final size hint will be (500, 400)
        """

        preferred_size = self._enable_window.component.get_preferred_size()
        q_size = self._enable_window.control.size()
        window_size = (q_size.width(), q_size.height())

        if qt_size_hint.width() < 0:
            width = max(preferred_size[0], window_size[0])
            qt_size_hint.setWidth(width)

        if qt_size_hint.height() < 0:
            height = max(preferred_size[1], window_size[1])
            qt_size_hint.setHeight(height)

        return qt_size_hint

    # ------------------------------------------------------------------------
    # Qt Drag and drop event handlers
    # ------------------------------------------------------------------------

    def dragEnterEvent(self, event):
        if self._enable_window:
            self._enable_window._drag_result = QtCore.Qt.IgnoreAction
            self._enable_window._handle_drag_event("drag_over", event)
            event.setDropAction(self._enable_window._drag_result)
            event.accept()

    def dragLeaveEvent(self, event):
        if self._enable_window:
            self._enable_window._handle_drag_event("drag_leave", event)

    def dragMoveEvent(self, event):
        if self._enable_window:
            self._enable_window._drag_result = QtCore.Qt.IgnoreAction
            self._enable_window._handle_drag_event("drag_over", event)
            event.setDropAction(self._enable_window._drag_result)
            event.accept()

    def dropEvent(self, event):
        if self._enable_window:
            self._enable_window._drag_result = event.proposedAction()
            self._enable_window._handle_drag_event("dropped_on", event)
            event.setDropAction(self._enable_window._drag_result)
            event.accept()


class _QtWindow(QtGui.QWidget):
    """ The Qt widget that implements the enable control. """

    def __init__(self, parent, enable_window):
        super(_QtWindow, self).__init__(parent)
        self.setAcceptDrops(True)
        self.handler = _QtWindowHandler(self, enable_window)

    def closeEvent(self, event):
        self.handler.closeEvent(event)
        return super(_QtWindow, self).closeEvent(event)

    def paintEvent(self, event):
        self.handler.paintEvent(event)

    def resizeEvent(self, event):
        self.handler.resizeEvent(event)

    def keyPressEvent(self, event):
        self.handler.keyPressEvent(event)

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

    def sizeHint(self):
        qt_size_hint = super(_QtWindow, self).sizeHint()
        return self.handler.sizeHint(qt_size_hint)


class _QtGLWindow(QtOpenGL.QGLWidget):
    def __init__(self, parent, enable_window):
        super(_QtGLWindow, self).__init__(parent)
        self.handler = _QtWindowHandler(self, enable_window)

    def closeEvent(self, event):
        self.handler.closeEvent(event)
        return super(_QtGLWindow, self).closeEvent(event)

    def paintEvent(self, event):
        super(_QtGLWindow, self).paintEvent(event)
        self.handler.paintEvent(event)
        self.swapBuffers()

    def resizeEvent(self, event):
        super(_QtGLWindow, self).resizeEvent(event)
        self.handler.resizeEvent(event)

    def keyPressEvent(self, event):
        self.handler.keyPressEvent(event)

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

    # TODO: by symmetry this belongs here, but we need to test it
    # def sizeHint(self):
    #    qt_size_hint = super(_QtGLWindow, self).sizeHint()
    #    return self.handler.sizeHint(qt_size_hint)


class _Window(AbstractWindow):

    control = Instance(QtGui.QWidget)

    def __init__(self, parent, wid=-1, pos=None, size=None, **traits):
        AbstractWindow.__init__(self, **traits)

        self._mouse_captured = False

        if isinstance(parent, QtGui.QLayout):
            parent = parent.parentWidget()
        self.control = self._create_control(parent, self)

        if self.high_resolution and hasattr(self.control, "devicePixelRatio"):
            self.base_pixel_scale = self.control.devicePixelRatio()

        if pos is not None:
            self.control.move(*pos)

        if size is not None:
            self.control.resize(*size)

    # ------------------------------------------------------------------------
    # Implementations of abstract methods in AbstractWindow
    # ------------------------------------------------------------------------

    def set_drag_result(self, result):
        if result not in DRAG_RESULTS_MAP:
            raise RuntimeError("Unknown drag result '%s'" % result)
        self._drag_result = DRAG_RESULTS_MAP[result]

    def _capture_mouse(self):
        "Capture all future mouse events"
        # Nothing needed with Qt.
        pass

    def _release_mouse(self):
        "Release the mouse capture"
        # Nothing needed with Qt.
        pass

    def _create_key_event(self, event_type, event):
        focus_owner = self.focus_owner

        if focus_owner is None:
            focus_owner = self.component

            if focus_owner is None:
                event.ignore()
                return None

        if event_type == "character":
            key = str(event.text())
        else:
            # Convert the keypress to a standard enable key if possible,
            # otherwise to text.
            key_code = event.key()
            key = KEY_MAP.get(key_code)
            if key is None:
                key = chr(key_code).lower()

        if not key:
            return None

        # Use the last-seen mouse position as the coordinates of this event.
        x, y = self.control.handler.last_mouse_pos

        modifiers = event.modifiers()

        return KeyEvent(
            event_type=event_type,
            character=key,
            x=x,
            y=self._flip_y(y),
            alt_down=bool(modifiers & QtCore.Qt.AltModifier),
            shift_down=bool(modifiers & QtCore.Qt.ShiftModifier),
            control_down=bool(modifiers & QtCore.Qt.ControlModifier),
            event=event,
            window=self,
        )

    def _create_mouse_event(self, event):
        # If the control no longer exists, don't send mouse event
        if self.control is None:
            return None
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

        self.control.handler.last_mouse_pos = (x, y)

        # A bit crap, because AbstractWindow was written with wx in mind, and
        # we treat wheel events like mouse events.
        if isinstance(event, QtGui.QWheelEvent):
            degrees_per_step = 15.0
            if is_qt4:
                delta = event.delta()
                mouse_wheel = delta / float(8 * degrees_per_step)
                mouse_wheel_axis = MOUSE_WHEEL_AXIS_MAP[event.orientation()]
                if mouse_wheel_axis == "horizontal":
                    mouse_wheel_delta = (delta, 0)
                else:
                    mouse_wheel_delta = (0, delta)
            else:
                delta = event.pixelDelta()
                if delta.x() == 0 and delta.y() == 0:  # pixelDelta is optional
                    delta = event.angleDelta()
                mouse_wheel_delta = (delta.x(), delta.y())
                if abs(mouse_wheel_delta[0]) > abs(mouse_wheel_delta[1]):
                    mouse_wheel = mouse_wheel_delta[0] / float(
                        8 * degrees_per_step
                    )
                    mouse_wheel_axis = "horizontal"
                else:
                    mouse_wheel = mouse_wheel_delta[1] / float(
                        8 * degrees_per_step
                    )
                    mouse_wheel_axis = "vertical"
        else:
            mouse_wheel = 0
            mouse_wheel_delta = (0, 0)
            mouse_wheel_axis = "vertical"

        return MouseEvent(
            x=x,
            y=self._flip_y(y),
            mouse_wheel=mouse_wheel,
            mouse_wheel_axis=mouse_wheel_axis,
            mouse_wheel_delta=mouse_wheel_delta,
            alt_down=bool(modifiers & QtCore.Qt.AltModifier),
            shift_down=bool(modifiers & QtCore.Qt.ShiftModifier),
            control_down=bool(modifiers & QtCore.Qt.ControlModifier),
            left_down=bool(buttons & QtCore.Qt.LeftButton),
            middle_down=bool(buttons & QtCore.Qt.MidButton),
            right_down=bool(buttons & QtCore.Qt.RightButton),
            window=self,
        )

    def _create_drag_event(self, event):

        # If the control no longer exists, don't send mouse event
        if self.control is None:
            return None
        # If the event (if there is one) doesn't contain the mouse position,
        # modifiers and buttons then get sensible defaults.
        try:
            x = event.x()
            y = event.y()
        except AttributeError:
            pos = self.control.mapFromGlobal(QtGui.QCursor.pos())
            x = pos.x()
            y = pos.y()

        self.control.handler.last_mouse_pos = (x, y)

        # extract an object from the event, if we can
        try:
            mimedata = event.mimeData()
            copy = event.proposedAction() == QtCore.Qt.CopyAction
        except AttributeError:
            # this is a DragLeave event
            return DragEvent(
                x=x,
                y=self._flip_y(y),
                obj=None,
                copy=False,
                window=self,
                mimedata=None,
            )

        try:
            from traitsui.qt4.clipboard import PyMimeData
        except ImportError:
            # traitsui isn't available, warn and just make mimedata available
            # on event
            warnings.warn("traitsui.qt4 is unavailable", ImportWarning)
            obj = None
        else:
            mimedata = PyMimeData.coerce(mimedata)
            obj = mimedata.instance()
            if obj is None:
                files = mimedata.localPaths()
                if files:
                    try:
                        # try to extract file info from mimedata
                        # XXX this is for compatibility with what wx does
                        from apptools.io.api import File

                        obj = [File(path=path) for path in files]
                    except ImportError:
                        warnings.warn("apptools is unavailable", ImportWarning)

        return DragEvent(
            x=x,
            y=self._flip_y(y),
            obj=obj,
            copy=copy,
            window=self,
            mimedata=mimedata,
        )

    def _redraw(self, coordinates=None):
        if self.control:
            if self.control.handler.in_paint_event:
                # Code further up the stack is behaving badly and calling
                # request_redraw() inside drawing code.
                return
            if coordinates is None:
                self.control.update()
            else:
                self.control.update(*coordinates)

    def _get_control_size(self):
        if self.control:
            return (int(self.control.width() * self.base_pixel_scale),
                    int(self.control.height() * self.base_pixel_scale))

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

    def _on_key_pressed(self, event):
        return self._handle_key_event("key_pressed", event)

    def get_pointer_position(self):
        pos = self.control.mapFromGlobal(QtGui.QCursor.pos())
        x = pos.x()
        y = self._flip_y(pos.y())
        return x, y

    # ------------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------------

    def _flip_y(self, y):
        """ Converts between a Kiva and a Qt y coordinate
        """
        # Handle the pixel scale adjustment here since `self._size` is involved
        return int(self._size[1] / self.base_pixel_scale - y - 1)


class BaseGLWindow(_Window):
    # The toolkit control
    control = Instance(_QtGLWindow)

    def _create_control(self, parent, enable_window):
        """ Create the toolkit control.
        """
        return _QtGLWindow(parent, enable_window)


class BaseWindow(_Window):
    # The toolkit control
    control = Instance(_QtWindow)

    def _create_control(self, parent, enable_window):
        """ Create the toolkit control.
        """
        return _QtWindow(parent, enable_window)
