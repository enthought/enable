

# Standard library imports

# Qt imports
from PyQt4.QtCore import Qt
from PyQt4.QtGui import QCursor, QImage, QPainter, QPixmap, QWheelEvent, QWidget

# Enthought library imports
from enthought.enable2.abstract_window import AbstractWindow
from enthought.enable2.events import KeyEvent, MouseEvent
from enthought.enable2.graphics_context import GraphicsContextEnable
from enthought.traits.api import Instance

# Relative imports
from constants import BUTTON_NAME_MAP, KEY_MAP, POINTER_MAP

WidgetClass = QWidget   # Should this be QFrame?

# TODO: Use the right Qt calls to establish these
DEFAULT_POSITION = (0,0)
DEFAULT_SIZE = (500,500)

class Window(AbstractWindow):
    
    control = Instance(WidgetClass)

    def __init__(self, parent, wid = -1, pos = DEFAULT_POSITION,
                 size = DEFAULT_SIZE, **traits):
        AbstractWindow.__init__(self, **traits)
        
        # Define some additional private attributes
        pos = QCursor.pos()
        self._last_mouse_pos = (pos.x(), pos.y())
        self._mouse_captured = False
        self._timer = None
        
        self.control = control = WidgetClass(parent, flags=0)
        # TODO: set nice default size?

        # Set some defaults to make it the control more amenable to Enable;
        # These mostly involve focus and such.
        control.setFocusPolicy(Qt.WheelFocus)
        control.mouseTracking = True
        return

    #------------------------------------------------------------------------
    # Qt window state and paint event handlers
    #------------------------------------------------------------------------

    def paintEvent(self, event):
        self._paint(event)
    
    def resizeEvent(self, event):
        dx, dy = self.control.size()
        self.resized = (dx, dy)     # Fire a trait event
        if hasattr(self.component, "fit_window") and self.component.fit_window:
            self.component.outer_position = [0,0]
            self.component.outer_bounds = [dx, dy]
        elif hasattr(self.component, "resizable"):
            if "h" in self.component.resizable:
                self.component.outer_x = 0
                self.component.outer_width = dx
            if "v" in self.component.resizable:
                self.component.outer_y = 0
                self.component.outer_height = dy
        
        # Is a single update() sufficient?  Do we need to do an invalidate, etc.?
        self.control.update()
        
    def hideEvent(self, event):
        pass
    
    def showEvent(self, event):
        pass
    
    def closeEvent(self, event):
        if self.component is not None:
            self.component.cleanup(self)
            self.component.parent = None
            self.component.window = None
            self.component = None
        if self.control is not None:
            self.control = None
        return
        
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
        x, y =  self._last_mouse_pos
        
        enable_event = KeyEvent(character = key,
                            alt_down = modifiers & Qt.AltModifier,
                            shift_down = modifiers & Qt.ShiftModifier,
                            control_down = modifiers & Qt.ControlModifier,
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

    def _capture_mouse ( self ):
        "Capture all future mouse events"
        # TODO: Investigate whether this is even necessary with Qt, since it
        # grabs the mouse if a key is pressed.
        if not self._mouse_captured:
            self._mouse_captured = True
            self.control.grabMouse()    
        return
    
    def _release_mouse ( self ):
        "Release the mouse capture"
        # TODO: see TODO in _capture_mouse()
        if self._mouse_captured:
            self._mouse_captured = False
            self.control.releaseMouse()
        return
    
    def _create_mouse_event(self, event):
        if event is not None:
            x = event.x()
            y = event.y()
            self._last_mouse_pos = (x,y)
            
            # A bit crap, because AbstractWindow was written with WX in mind, and
            # we treat wheel events like mouse events
            if isinstance(event, QWheelEvent):
                delta = event.delta()
                degrees_per_step = 15.0
                mouse_wheel = delta / float(8 * degrees_per_step)
            else:
                mouse_wheel = 0
            
            modifiers = event.modifiers()
            buttons = event.buttons()
        
        else:
            # If no event was specified, make one up
            pos = self.control.mapFromGlobal(QCursor.pos())
            x = pos.x()
            y = pos.y()
            self._last_mouse_pos = (x,y)
            mouse_wheel = 0
            modifiers = 0
            buttons = 0

        return MouseEvent(x=x, y=self._flip_y(y), mouse_wheel=mouse_wheel,
                        alt_down = modifiers & Qt.AltModifier,
                        shift_down = modifiers & Qt.ShiftModifier,
                        control_down = modifiers & Qt.ControlModifier,
                        left_down = buttons & Qt.LeftButton,
                        middle_down = buttons & Qt.MiddleButton,
                        right_down = buttons & Qt.RightButton,
                        window = self)

    def _redraw(self, coordinates=None):
        if not self.control:
            return
        if coordinates is None:
            self.control.update()
        else:
            self.control.update(*coordinates)
        return
    
    def _get_control_size(self):
        if self.control:
            dx, dy = self.control.size()
            return (dx, dy)
        else:
            return None
    
    def _create_gc(self, size, pix_format="bgr24"):
        gc = GraphicsContextEnable((size[0]+1, size[1]+1), pix_format=pix_format,
                                    window=self)
        gc.translate_ctm(0.5, 0.5)
        return gc
    
    def _window_paint(self, event):
        s = self._gc.pixel_map.convert_to_argb32string()
        img = QImage(s, self.gc.width(), self.gc.height(),
                     QImage.Format_ARGB32)
        p = QPainter(self)
        p.drawPixmap(0, 0, QPixmap.fromImage(img))
    
    def set_pointer(self, pointer):
        ptr = POINTER_MAP[pointer]
        self.control.SetCursor(ptr)
    
    def set_tooltip(self, tooltip):
        pass
    
    def _set_focus(self):
        self.control.setFocus(Qt.OtherFocusReason)
    
    #------------------------------------------------------------------------
    # Private methods
    #------------------------------------------------------------------------

    def _flip_y(self, y):
        "Converts between a Kiva and a Qt y coordinate"
        return int(self._size[1] - y - 1)
