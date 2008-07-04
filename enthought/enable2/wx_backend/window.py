"""
Defines the concrete top-level Enable 'Window' class for the wxPython GUI 
toolkit, based on the kiva agg driver.
"""

import sys
import time
import wx


from enthought.traits.api import Any, Instance, Trait
from enthought.traits.ui.wx.menu import MakeMenu

# Relative imports
from enthought.enable2.base import union_bounds
from enthought.enable2.component  import Component
from enthought.enable2.events import MouseEvent, KeyEvent, DragEvent
from enthought.enable2.graphics_context import GraphicsContextEnable
from enthought.enable2.abstract_window import AbstractWindow
from enthought.kiva import backend

if backend() == "gl":
    from wx.glcanvas import GLCanvas
    WidgetClass = GLCanvas
else:
    WidgetClass = wx.Window

try:
    from enthought.util.wx.drag_and_drop import clipboard, PythonDropTarget
except ImportError:
    clipboard = None
    PythonDropTarget = None

#-------------------------------------------------------------------------------
#  Constants:
#-------------------------------------------------------------------------------

# Number of pixels to scroll at a time:
scroll_incr = 16

# Map from pointer shape name to pointer shapes:
pointer_map = {
   'arrow':             wx.CURSOR_ARROW,
   'right arrow':       wx.CURSOR_RIGHT_ARROW,
   'blank':             wx.CURSOR_BLANK,
   'bullseye':          wx.CURSOR_BULLSEYE,
   'char':              wx.CURSOR_CHAR,
   'cross':             wx.CURSOR_CROSS,
   'hand':              wx.CURSOR_HAND,
   'ibeam':             wx.CURSOR_IBEAM,
   'left button':       wx.CURSOR_LEFT_BUTTON,
   'magnifier':         wx.CURSOR_MAGNIFIER,
   'middle button':     wx.CURSOR_MIDDLE_BUTTON,
   'no entry':          wx.CURSOR_NO_ENTRY,
   'paint brush':       wx.CURSOR_PAINT_BRUSH,
   'pencil':            wx.CURSOR_PENCIL,
   'point left':        wx.CURSOR_POINT_LEFT,
   'point right':       wx.CURSOR_POINT_RIGHT,
   'question arrow':    wx.CURSOR_QUESTION_ARROW, 
   'right button':      wx.CURSOR_RIGHT_BUTTON, 
   'size top':          wx.CURSOR_SIZENS, 
   'size bottom':       wx.CURSOR_SIZENS, 
   'size left':         wx.CURSOR_SIZEWE, 
   'size right':        wx.CURSOR_SIZEWE, 
   'size top right':    wx.CURSOR_SIZENESW, 
   'size bottom left':  wx.CURSOR_SIZENESW, 
   'size top left':     wx.CURSOR_SIZENWSE, 
   'size bottom right': wx.CURSOR_SIZENWSE, 
   'sizing':            wx.CURSOR_SIZING, 
   'spray can':         wx.CURSOR_SPRAYCAN,
   'wait':              wx.CURSOR_WAIT, 
   'watch':             wx.CURSOR_WATCH,
   'arrow wait':        wx.CURSOR_ARROWWAIT
}

# Map from wxPython special key names into Enable key names:
key_map = {
    wx.WXK_BACK:      'Backspace',
    wx.WXK_TAB:       'Tab',
    wx.WXK_RETURN:    'Enter',
    wx.WXK_ESCAPE:    'Esc',
    wx.WXK_DELETE:    'Delete',
    wx.WXK_START:     'Start',
    wx.WXK_LBUTTON:   'Left Button',
    wx.WXK_RBUTTON:   'Right Button',
    wx.WXK_CANCEL:    'Cancel',
    wx.WXK_MBUTTON:   'Middle Button',
    wx.WXK_CLEAR:     'Clear',
    wx.WXK_SHIFT:     'Shift',
    wx.WXK_CONTROL:   'Control',
    wx.WXK_MENU:      'Menu',
    wx.WXK_PAUSE:     'Pause',
    wx.WXK_CAPITAL:   'Capital',
    wx.WXK_PRIOR:     'Page Up',
    wx.WXK_NEXT:      'Page Down',
    wx.WXK_END:       'End',
    wx.WXK_HOME:      'Home',
    wx.WXK_LEFT:      'Left',
    wx.WXK_UP:        'Up',
    wx.WXK_RIGHT:     'Right',
    wx.WXK_DOWN:      'Down',
    wx.WXK_SELECT:    'Select',
    wx.WXK_PRINT:     'Print',
    wx.WXK_EXECUTE:   'Execute',
    wx.WXK_SNAPSHOT:  'Snapshot',
    wx.WXK_INSERT:    'Insert',
    wx.WXK_HELP:      'Help',
    wx.WXK_NUMPAD0:   'Numpad 0',
    wx.WXK_NUMPAD1:   'Numpad 1',
    wx.WXK_NUMPAD2:   'Numpad 2',
    wx.WXK_NUMPAD3:   'Numpad 3',
    wx.WXK_NUMPAD4:   'Numpad 4',
    wx.WXK_NUMPAD5:   'Numpad 5',
    wx.WXK_NUMPAD6:   'Numpad 6',
    wx.WXK_NUMPAD7:   'Numpad 7',
    wx.WXK_NUMPAD8:   'Numpad 8',
    wx.WXK_NUMPAD9:   'Numpad 9',
    wx.WXK_MULTIPLY:  'Multiply',
    wx.WXK_ADD:       'Add',
    wx.WXK_SEPARATOR: 'Separator',
    wx.WXK_SUBTRACT:  'Subtract',
    wx.WXK_DECIMAL:   'Decimal',
    wx.WXK_DIVIDE:    'Divide',
    wx.WXK_F1:        'F1',
    wx.WXK_F2:        'F2',
    wx.WXK_F3:        'F3',
    wx.WXK_F4:        'F4',
    wx.WXK_F5:        'F5',
    wx.WXK_F6:        'F6',
    wx.WXK_F7:        'F7',
    wx.WXK_F8:        'F8',
    wx.WXK_F9:        'F9',
    wx.WXK_F10:       'F10',
    wx.WXK_F11:       'F11',
    wx.WXK_F12:       'F12',
    wx.WXK_F13:       'F13',
    wx.WXK_F14:       'F14',
    wx.WXK_F15:       'F15',
    wx.WXK_F16:       'F16',
    wx.WXK_F17:       'F17',
    wx.WXK_F18:       'F18',
    wx.WXK_F19:       'F19',
    wx.WXK_F20:       'F20',
    wx.WXK_F21:       'F21',
    wx.WXK_F22:       'F22',
    wx.WXK_F23:       'F23',
    wx.WXK_F24:       'F24',
    wx.WXK_NUMLOCK:   'Num Lock',
    wx.WXK_SCROLL:    'Scroll Lock'
}

drag_results_map = { "error": wx.DragError,
                     "none": wx.DragNone,
                     "copy": wx.DragCopy,
                     "move": wx.DragMove,
                     "link": wx.DragLink,
                     "cancel": wx.DragCancel }



# Reusable instance to avoid constructor/destructor overhead:
wx_rect = wx.Rect( 0, 0, 0, 0 )

# Default 'fake' start event for wxPython based drag operations:
default_start_event = MouseEvent()


# To conserve system resources, there is only one 'timer' per program:
system_timer = None
 
class EnableTimer ( wx.Timer ):
    """
    This class maintains a 'work list' of scheduled components, where 
    each item in the list has the form: [ component, interval, timer_pop_time ]
    """
    
    def __init__ ( self ):
        wx.Timer.__init__( self )
        self._work_list = []
        return
    
    def schedule ( self, component, interval ):
        "Schedule a timer event for a specified component"
        work_list = self._work_list
        if len( work_list ) == 0:
            self.Start( 5, oneShot=False )
        for i, item in enumerate( work_list ):
            if component is item[0]:
                del work_list[i]
                break
        self.reschedule( component, interval )
        return

    def reschedule ( self, component, interval ):
        "Reshedule a recurring timer event for a component"
        pop_time  = time.time() + interval
        new_item  = [ component, interval, pop_time ]
        work_list = self._work_list
        for i, item in enumerate( work_list ):
            if pop_time < item[2]:
                work_list.insert( i, new_item )
                break
        else:
            work_list.append( new_item )
        return
    
    def cancel ( self, component ):
        "Cancel any pending timer events for a component"
        work_list = self._work_list
        for i, item in enumerate( work_list ):
            if component is item[0]:
                del work_list[i]
                if len( work_list ) == 0:
                    self.Stop()
                break
        return (len( work_list ) != 0)        
        
    def Notify ( self ):
        "Handle a timer 'pop' event; used for performance testing purposes"
        now       = time.time()
        work_list = self._work_list
        n         = len( work_list )
        i         = 0
        while (i < n) and (now >= work_list[i][2]):
            i += 1
        if i > 0:
            reschedule = work_list[:i]
            del work_list[:i]
            for component, interval, ignore in reschedule:
                self.reschedule( component, interval )
                component.timer = True
        return


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
        
        if clipboard.drop_source is not None and \
           not clipboard.drop_source.allow_move:
            default_drag_result = wx.DragCopy

        if hasattr(self.handler, 'wx_drag_over'):
            drag_result = self.handler.wx_drag_over(
                x, y, clipboard.data, default_drag_result
            )
        else:
            drag_result = default_drag_result

        return drag_result


class Window ( AbstractWindow ):

    # Screen scroll increment amount:
    scroll_incr = ( wx.SystemSettings_GetMetric( wx.SYS_SCREEN_Y )
                    or 768 ) / 20

    # Width/Height of standard scrollbars:
    scrollbar_dx = wx.SystemSettings_GetMetric( wx.SYS_VSCROLL_X )
    scrollbar_dy = wx.SystemSettings_GetMetric( wx.SYS_HSCROLL_Y )

    _cursor_color = Any  # PZW: figure out the correct type for this...

    # Reference to the actual wxPython window:
    control      = Instance(WidgetClass)
    
    # This is set by downstream components to notify us of whether or not
    # the current drag operation should return DragCopy, DragMove, or DragNone.
    _drag_result = Any
    
    def __init__ ( self, parent, wid = -1, pos = wx.DefaultPosition,
                   size = wx.DefaultSize, **traits ):
        AbstractWindow.__init__( self, **traits )
        self._timer          = None
        self._mouse_captured = False

        # If we are using the GL backend, we will need to have a pyglet
        # GL context
        self._pyglet_gl_context = None

        # Due to wx wonkiness, we don't reliably get cursor position from
        # a wx KeyEvent.  Thus, we manually keep track of when we last saw
        # the mouse and use that information instead.  These coordinates are
        # in the wx coordinate space, i.e. pre-self._flip_y().
        self._last_mouse_pos = (0, 0)
        
        # Create the delegate: 
        self.control = control = WidgetClass( parent, wid, pos, size,
                                              style = wx.CLIP_CHILDREN |
                                                      wx.WANTS_CHARS )
        
        # Set up the 'erase background' event handler:
        wx.EVT_ERASE_BACKGROUND( control, self._on_erase_background )
 
        # Set up the 'paint' event handler:
        wx.EVT_PAINT( control, self._paint )
        wx.EVT_SIZE(  control, self._on_size )
        
        # Set up mouse event handlers:
        wx.EVT_LEFT_DOWN(     control, self._on_left_down )
        wx.EVT_LEFT_UP(       control, self._on_left_up )
        wx.EVT_LEFT_DCLICK(   control, self._on_left_dclick )
        wx.EVT_MIDDLE_DOWN(   control, self._on_middle_down )
        wx.EVT_MIDDLE_UP(     control, self._on_middle_up )
        wx.EVT_MIDDLE_DCLICK( control, self._on_middle_dclick )
        wx.EVT_RIGHT_DOWN(    control, self._on_right_down )
        wx.EVT_RIGHT_UP(      control, self._on_right_up )
        wx.EVT_RIGHT_DCLICK(  control, self._on_right_dclick )
        wx.EVT_MOTION(        control, self._on_mouse_move )
        wx.EVT_ENTER_WINDOW(  control, self._on_window_enter )
        wx.EVT_LEAVE_WINDOW(  control, self._on_window_leave )
        wx.EVT_MOUSEWHEEL(    control, self._on_mouse_wheel )
        
        # Handle key up/down events:
        wx.EVT_KEY_DOWN( control, self._on_key_updown )
        wx.EVT_KEY_UP(   control, self._on_key_updown )
        wx.EVT_CHAR(     control, self._on_char )
            
        # Attempt to allow wxPython drag and drop events to be mapped to
        # Enable drag events:
        
        # Handle window close and cleanup
        wx.EVT_WINDOW_DESTROY(control, self._on_close)

        if PythonDropTarget is not None:
            control.SetDropTarget( LessSuckyDropTarget( self ) ) 
            self._drag_over = []
        return
           
    def _on_close(self, event):
        # Might be scrollbars or other native components under
        # us that are generating this event
        if event.GetWindow() == self.control:
            self._gc = None
            wx.EVT_ERASE_BACKGROUND(self.control, None)
            wx.EVT_PAINT(self.control, None)
            wx.EVT_SIZE(self.control, None)
            wx.EVT_LEFT_DOWN(self.control, None)
            wx.EVT_LEFT_UP(self.control, None)
            wx.EVT_LEFT_DCLICK(self.control, None)
            wx.EVT_MIDDLE_DOWN(self.control, None)
            wx.EVT_MIDDLE_UP(self.control, None)
            wx.EVT_MIDDLE_DCLICK(self.control, None)
            wx.EVT_RIGHT_DOWN(self.control, None)
            wx.EVT_RIGHT_UP(self.control, None)
            wx.EVT_RIGHT_DCLICK(self.control, None)
            wx.EVT_MOTION(self.control, None)
            wx.EVT_ENTER_WINDOW(self.control, None)
            wx.EVT_LEAVE_WINDOW(self.control, None)
            wx.EVT_MOUSEWHEEL(self.control, None)
            wx.EVT_KEY_DOWN(self.control, None)
            wx.EVT_KEY_UP(self.control, None)
            wx.EVT_CHAR(self.control, None)
            wx.EVT_WINDOW_DESTROY(self.control, None)
            self.control.SetDropTarget(None)
            self.control = None
            self.component.cleanup(self)
            self.component.parent = None
            self.component.window = None
            self.component = None
        return
        
    def _on_key_updown ( self, event ):
        "Handle keyboard keys changing their up/down state"
        k = event.GetKeyCode()
        t = event.GetEventType()

        if k == wx.WXK_SHIFT:
            self.shift_pressed = (t == wx.wxEVT_KEY_DOWN)
        elif k == wx.WXK_ALT:
            self.alt_pressed   = (t == wx.wxEVT_KEY_DOWN)
        elif k == wx.WXK_CONTROL:
            self.ctrl_pressed  = (t == wx.wxEVT_KEY_DOWN)

        event.Skip()
        return

    def _on_char ( self, event ):
        "Handle keyboard keys being pressed"

        if self.focus_owner is None:
            focus_owner = self.component
        else:
            focus_owner = self.focus_owner
        
        if focus_owner is not None:
            control_down = event.ControlDown()
            key_code     = event.GetKeyCode()
            key = None
            if control_down:
                if (1 <= key_code <= 26):
                    key = chr( key_code + 96 )
            elif key_map.has_key(key_code):
                key = key_map.get( key_code )
            if key is None:
                if key_code >= 0 and key_code < 256:
                    key = chr( key_code )

            # Use the last-seen mouse coordinates instead of GetX/GetY due
            # to wx bug.
            x, y = self._last_mouse_pos

            # Someday when wx does this properly, we can use these instead:
            # x = event.GetX()
            # y = event.GetY()
            
            enable_event = KeyEvent( character = key,
                                     alt_down = event.AltDown(),
                                     control_down = control_down,
                                     shift_down = event.ShiftDown(),
                                     x = x,
                                     y = self._flip_y(y),
                                     event = event,
                                     window = self )
            focus_owner.dispatch(enable_event, "key_pressed")
        else:
            event.Skip()
        return
    
    def _flip_y ( self, y ):
        "Convert from a Kiva to a wxPython y coordinate"
        return int( self._size[1] - 1 - y )
   
    def _on_erase_background ( self, event ):
        pass

    def _on_size ( self, event ):
        dx, dy = self.control.GetSizeTuple()

        # do nothing if the new and old sizes are the same
        if (self.component.outer_width, self.component.outer_height) == (dx, dy):
            return
        
        self.resized = (dx, dy)
        
        if getattr(self.component, "fit_window", False):
            self.component.outer_position = [0,0]
            self.component.outer_bounds = [dx, dy]
        elif hasattr(self.component, "resizable"):
            if "h" in self.component.resizable:
                self.component.outer_x = 0
                self.component.outer_width = dx
            if "v" in self.component.resizable:
                self.component.outer_y = 0
                self.component.outer_height = dy
        
        self.control.Refresh()
        return
    
    def _capture_mouse ( self ):
        "Capture all future mouse events"
        if not self._mouse_captured:
            self._mouse_captured = True
            self.control.CaptureMouse()    
        return
    
    def _release_mouse ( self ):
        "Release the mouse capture"
        if self._mouse_captured:
            self._mouse_captured = False
            self.control.ReleaseMouse()
        return
    
    def _create_mouse_event ( self, event ):
        "Convert a GUI toolkit mouse event into a MouseEvent"
        if event is not None:
            x           = event.GetX()
            y           = event.GetY()
            self._last_mouse_pos = (x, y)
            mouse_wheel = ((event.GetLinesPerAction() * 
                            event.GetWheelRotation()) / 
                            (event.GetWheelDelta() or 1))
            
            # Note: The following code fixes a bug in wxPython that returns
            # 'mouse_wheel' events in screen coordinates, rather than window
            # coordinates:
            if float(wx.VERSION_STRING[:3]) < 2.8:
                if mouse_wheel != 0 and sys.platform == "win32":
                    x, y = self.control.ScreenToClientXY( x, y )
            return MouseEvent( x            = x,
                               y            = self._flip_y( y ),
                               alt_down     = event.AltDown(),    
                               control_down = event.ControlDown(),
                               shift_down   = event.ShiftDown(),
                               left_down    = event.LeftIsDown(),
                               middle_down  = event.MiddleIsDown(),
                               right_down   = event.RightIsDown(),
                               mouse_wheel  = mouse_wheel,
                               window = self )
                               
        # If no event specified, make one up:
        x, y = wx.GetMousePosition()
        x, y = self.control.ScreenToClientXY( x, y )
        self._last_mouse_pos = (x, y)
        return MouseEvent( x            = x,
                           y            = self._flip_y( y ),
                           alt_down     = self.alt_pressed,    
                           control_down = self.ctrl_pressed,
                           shift_down   = self.shift_pressed,
                           left_down    = False,
                           middle_down  = False,
                           right_down   = False,
                           mouse_wheel  = 0,
                           window = self)
    
    def _create_gc ( self, size, pix_format = "bgra32" ):
        "Create a Kiva graphics context of a specified size"
        gc = GraphicsContextEnable((size[0]+1, size[1]+1), pix_format = pix_format, window=self )
        gc.translate_ctm(0.5, 0.5)
        return gc
    
    def _redraw(self, coordinates=None):
        "Request a redraw of the window"
        if coordinates is None:
            if self.control:
                self.control.Refresh( False )
        else:
            xl, yb, xr, yt = coordinates
            rect = wx_rect
            rect.SetX( int( xl ) )
            rect.SetY( int( self._flip_y( yt - 1 ) ) )
            rect.SetWidth(  int( xr - xl ) )
            rect.SetHeight( int( yt - yb ) )
            if self.control:
                self.control.Refresh( False, rect )
        return
    
    def _get_control_size ( self ):
        "Get the size of the underlying toolkit control"
        result = None
        if self.control:
            result = self.control.GetSizeTuple()
        return result

    def _init_gc(self):
        if backend() == "gl":
            gc = GraphicsContextEnable(self._size, window=self)
            if self._pyglet_gl_context is None:
                from pyglet.gl import Context
                self._pyglet_gl_context = Context()
            self._pyglet_gl_context.set_current()
            self.control.SetCurrent()
            gc.gl_init()
            self._gc = gc
        else:
            gc = self._gc
        gc.clear(self.bg_color_)

    def _window_paint ( self, event):
        "Do a GUI toolkit specific screen update"
        if backend() == "gl":
            self.control.SwapBuffers()
        else:
            control = self.control
            wdc     = control._dc = wx.PaintDC( control )
            self._update_region = None
            if self._update_region is not None:
                update_bounds = reduce(union_bounds, self._update_region)
                self._gc.pixel_map.draw_to_wxwindow( control, int(update_bounds[0]), int(update_bounds[1]),
                                                     width=int(update_bounds[2]), height=int(update_bounds[3]))
            else:
                self._gc.pixel_map.draw_to_wxwindow( control, 0, 0 )
            control._dc = None
        return
        
    def set_pointer ( self, pointer ):
        "Set the current pointer (i.e. cursor) shape"
        ptr = pointer_map[ pointer ]
        if type( ptr ) is int:
            pointer_map[ pointer ] = ptr = wx.StockCursor( ptr )
        self.control.SetCursor( ptr )
        return
        
    def set_tooltip ( self, tooltip ):
        "Set the current tooltip for the window"
        wx.ToolTip_Enable( False )
        self.control.SetToolTip( wx.ToolTip( tooltip ) )
        wx.ToolTip_Enable( True )
        return

    def set_timer_interval ( self, component, interval ):
        """ Set up or cancel a timer for a specified component.  To cancel the
        timer, set interval=None """
        global system_timer
        if interval is None:
            if ((system_timer is not None) and 
                (not system_timer.cancel( component ))):
                system_timer = None
        else:
            if system_timer is None:
                system_timer = EnableTimer()
            system_timer.schedule( component, interval )
        return
    
    def _set_focus ( self ):
        "Sets the keyboard focus to this window"
        self.control.SetFocus()
        return

    def screen_to_window(self, x, y):
        pt = wx.Point(x,y)
        x,y = self.control.ScreenToClient(pt)
        y = self._flip_y(y)
        return x,y
    
    def set_drag_result(self, result):
        if result not in drag_results_map:
            raise RuntimeError, "Unknown drag result '%s'" % result
        self._drag_result = drag_results_map[result]
        return

    def wx_dropped_on ( self, x, y, drag_object, drop_result ):
        "Handle wxPython drag and drop events"
        # Process the 'dropped_on' event for the object(s) it was dropped on:
        y = self._flip_y(y)
        drag_event = DragEvent(x=x, y=y, obj=drag_object, window=self)
        self._drag_result = wx.DragNone
        if self.component.is_in(x, y):
            self.component.dispatch(drag_event, "dropped_on")
        
        # If a downstream component wants to express that it handled the 
        return self._drag_result

    def wx_drag_over ( self, x, y, drag_object, drag_result ):
        y = self._flip_y( y )
        drag_over_event = DragEvent( x    = x,
                                     y    = y,
                                     x0   = 0.0,
                                     y0   = 0.0,
                                     copy = drag_result != wx.DragMove,
                                     obj = drag_object,
                                     start_event = default_start_event,
                                     window = self )
        
        # By default, don't indicate that we can be dropped on.  It is up
        # to the component to set this correctly.
        self._drag_result = wx.DragNone
        if self.component.is_in(x, y):
            self.component.dispatch(drag_over_event, "drag_over")
        
        return self._drag_result
        
    def wx_drag_leave ( self, drag_object ):
        drag_leave_event = DragEvent( x    = 0.0,
                                     y    = 0.0,
                                     x0   = 0.0,
                                     y0   = 0.0,
                                     copy = False,
                                     obj = drag_object,
                                     start_event = default_start_event,
                                     window = self )
        self.component.dispatch(drag_leave_event, "drag_leave")
        return
    
    def create_menu ( self, menu_definition, owner ):
        "Create a wxMenu from a string description"
        return MakeMenu( menu_definition, owner, True, self.control )
    
    def popup_menu ( self, menu, x, y ):
        "Pop-up a wxMenu at a specified location"
        self.control.PopupMenuXY( menu.menu, int(x), int( self._flip_y(y) ) ) 
        return


if wx.__version__.startswith("2.6"):
    WxWindowTrait = Trait(None, wx.WindowPtr)
elif wx.__version__.startswith("2.8"):
    WxWindowTrait = Trait(None, wx.Window)
else:
    raise RuntimeError("Unsupported wxPython version.")

class WindowComponent ( Component ):
    
    component = WxWindowTrait
    
    def __init__ ( self, component = None, **traits ):
        Component.__init__( self, **traits )
        if component is not None:
            self.component  = component
            size            = component.GetAdjustedBestSize()
            self.min_width  = size.GetWidth()
            self.min_height = size.GetHeight()
            self._container_changed( self.container )
            
    def _is_valid ( self ):
        "Determine if it is valid to manipulate the wxWindow component"
        return ((self.component is not None) and 
                isinstance( self.container.window, Window ))
                
    def _set_visibility ( self ):
        "Set the correct visibility for the wxWindows component"
        self.component.Show( 
                 isinstance( self.container.window, Window ) and self.visible )
        return
        
    def _container_changed ( self, container ):
        "Handle the container being changed"
        if self.component is not None:
            self._set_visibility()
            self._bounds_changed( self.bounds )
        return
    
    def _bounds_changed ( self, bounds ):
        "Handle the bounds of the component being changed"
        if self.component is not None:
            x, y, dx, dy = bounds
            comp = self.component
            comp.SetDimensions( 
                 int( x ), int( comp.GetParent()._flip_y( y + dy - 1.0 ) ), 
                 int( dx ), int( dy ) )
            comp.Refresh()
        return
       
    def _components_at ( self, x, y ):
        "Generate any additional components that contain a specified (x,y) point"
        return [ self ]
        
    def draw ( self, gc ):
        "Draw the component in a specified graphics context if it needs it"
        if self._drawable and self.window._needs_redraw( self.bounds ):
            comp = self.component
            if comp is not None:
                comp.Show( self.visible )
                comp.Refresh()
        return
    
# EOF
