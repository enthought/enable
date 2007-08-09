"""
Defines the concrete top-level Window class for the Tkinter GUI toolkit, 
based on the Kiva agg driver and Tkinter 3000.
"""
from WCK import Widget, Controller

# Relative imports
from enthought.enable.abstract_window import AbstractWindow
from enthought.enable.events import MouseEvent
from enthought.enable.graphics_context import GraphicsContextEnable

# Map from Enable pointer names to Tkinter pointer names:
pointer_map = {
   'arrow':             '',
   'right arrow':       'arrow',
   'blank':             'gumby',
   'bullseye':          'target',
   'char':              'xterm',
   'cross':             'crosshair',
   'hand':              'hand1',
   'ibeam':             'xterm',
   'left button':       'leftbutton',
   'magnifier':         'circle',
   'middle button':     'middlebutton',
   'no entry':          'X_cursor',
   'paint brush':       'gumby',
   'pencil':            'pencil',
   'point left':        'sb_left_arrow',
   'point right':       'sb_right_arrow',
   'question arrow':    'question_arrow',
   'right button':      'rightbutton',
   'size top':          'top_side',
   'size bottom':       'bottom_side',
   'size left':         'left_side',
   'size right':        'right_side',
   'size top right':    'top_right_corner',
   'size bottom left':  'bottom_left_corner',
   'size top left':     'top_left_corner',
   'size bottom right': 'bottom_right_corner',
   'sizing':            'sizing',
   'spray can':         'spraycan',
   'wait':              'wait',
   'watch':             'watch',
   'arrow wait':        'watch'
}     

#-------------------------------------------------------------------------------
#  'KivaController' class:
#-------------------------------------------------------------------------------

class KivaController ( Controller ):

    active = None
    inside = None

    def create ( self, bind ):        
#        # Set up mouse event handlers:
#        wx.EVT_MOUSEWHEEL(    self, self._on_mouse_wheel )
        
        # !! Only a portion of the events are bound...
        bind( "<Button-1>",         self._on_left_down )
        bind( "<Button-2>",         self._on_middle_down )
        bind( "<Button-3>",         self._on_right_down )
        bind( "<Double-Button-1>",  self._on_left_dclick )
        bind( "<Double-Button-2>",  self._on_middle_dclick )
        bind( "<Double-Button-3>",  self._on_right_dclick )
        bind( "<ButtonRelease-1>",  self._on_left_up )
        bind( "<ButtonRelease-2>",  self._on_middle_up )
        bind( "<ButtonRelease-3>",  self._on_right_up )
        bind( "<Motion>",           self._on_mouse_move )
        bind( "<Enter>",            self._on_window_enter )
        bind( "<Leave>",            self._on_window_leave )

    def _on_left_down ( self, event ):
        event.widget._enable_window._on_left_down( event )

    def _on_middle_down ( self, event ):
        event.widget._enable_window._on_middle_down( event )

    def _on_right_down ( self, event ):
        event.widget._enable_window._on_right_down( event )

    def _on_left_dclick ( self, event ):
        event.widget._enable_window._on_left_dclick( event )

    def _on_middle_dclick ( self, event ):
        event.widget._enable_window._on_middle_dclick( event )

    def _on_right_dclick ( self, event ):
        event.widget._enable_window._on_right_dclick( event )
        
    def _on_left_up ( self, event ):
        event.widget._enable_window._on_left_up( event )
        
    def _on_middle_up ( self, event ):
        event.widget._enable_window._on_middle_up( event )
        
    def _on_right_up ( self, event ):
        event.widget._enable_window._on_right_up( event )

    def _on_mouse_move ( self, event ):
        event.widget._enable_window._on_mouse_move( event )

    def _on_window_enter ( self, event ):
        event.widget._enable_window._on_window_enter( event )

    def _on_window_leave ( self, event ):
        event.widget._enable_window._on_window_leave( event )

#-------------------------------------------------------------------------------
#  'EnableTkWindow' class:
#-------------------------------------------------------------------------------

class EnableTkWindow ( AbstractWindow ):

    def __init__ ( self, tk_widget, **traits ):
        self._tk_widget = tk_widget
        AbstractWindow.__init__( self, **traits )
        
    #---------------------------------------------------------------------------
    #  Convert a GUI toolkit mouse event into a MouseEvent:
    #---------------------------------------------------------------------------
     
    def _create_mouse_event ( self, event ):
        state = event.state
        num   = event.num
        return MouseEvent( x            = event.x,           
                           y            = self._flip_y( event.y ),
                           alt_down     = (state & 0x20000) != 0,
                           control_down = (state & 4) != 0,
                           shift_down   = (state & 1) != 0,
                           left_down    = (num == 1) or ((state & 0x0100) != 0),
                           middle_down  = (num == 2) or ((state & 0x0200) != 0),
                           right_down   = (num == 3) or ((state & 0x0400) != 0))
        
    #---------------------------------------------------------------------------
    #  Capture all future mouse events:
    #---------------------------------------------------------------------------
    
    def _capture_mouse ( self ):
        self._tk_widget.grab_set()    

    #---------------------------------------------------------------------------
    #  Release the mouse capture:
    #---------------------------------------------------------------------------
    
    def _release_mouse ( self ):
        self._tk_widget.grab_release()

    #---------------------------------------------------------------------------
    #  Create a Kiva graphics context of a specified size:  
    #---------------------------------------------------------------------------
    
    def _create_gc ( self, size, pix_format = "bgr24" ):
        gc = GraphicsContextEnable( size )
        gc._clip_bounds = []   ### TEMPORARY ###
        return gc
            
    #---------------------------------------------------------------------------
    #  Request a redraw of the window:
    #---------------------------------------------------------------------------
    
    def _redraw ( self, coordinates ):
        if coordinates is None:
            self._tk_widget.ui_damage()
        else:
            xl, yb, xr, yt = coordinates
            self._tk_widget.ui_damage( int( xl ), int( self._flip_y( yt - 1 ) ),
                                       int( xr ), int( self._flip_y( yb ) ) )
        
    #---------------------------------------------------------------------------
    #  Get the appropriate gc size for a specified component size:
    #---------------------------------------------------------------------------
    
    def _get_gc_size ( self, size ):
        return size
        
    #---------------------------------------------------------------------------
    #  Set the size of the associated component:
    #---------------------------------------------------------------------------
    
    def _set_component_size ( self, size ):
        pass
               
    #---------------------------------------------------------------------------
    #  Set the current pointer (i.e. cursor) shape:
    #---------------------------------------------------------------------------
        
    def _set_pointer ( self, pointer ):
        self._tk_widget.config( cursor = pointer_map[ pointer ] )
        
    #---------------------------------------------------------------------------
    #  Convert from a Kiva to a wxPython y coordinate: 
    #---------------------------------------------------------------------------

    def _flip_y ( self, y ):
        return self._tk_widget.ui_size()[1] - 1 - y

    def _window_paint ( self, event ):
        # NOTE: This should do an optimal update based on the value of the
        # self._update_region, but Kiva does not currently support this:
        self._gc.pixel_map.draw_to_tkwindow( self, 0, 0 )

#----------------------------------------------------------------------------
#  'Window' class:
#----------------------------------------------------------------------------
       
class Window ( Widget ):

    ui_controller     = KivaController
    ui_option_command = None

    def __init__ ( self, parent, **traits ):
        Widget.__init__( self, parent )
        self._enable_window = EnableTkWindow( self, **traits )

    #---------------------------------------------------------------------------
    #  Paint related events
    #---------------------------------------------------------------------------

    def ui_handle_clear ( self, draw, x0, y0, x1, y1 ):
        pass

    def ui_handle_resize ( self, width, height ):       
        if self._enable_window.component:
            self._enable_window.component.bounds = ( 0.0, 0.0, 
                                      float( width ), float( height ) )  

    def ui_handle_damage ( self, x0, y0, x1, y1 ):
        pass
        
    def ui_handle_repair ( self, draw, x0, y0, x1, y1 ):
        self._draw = draw
        self._enable_window._paint()
    
