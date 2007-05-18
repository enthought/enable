
# Major library imports
from types import ListType

# Enthought library imports
from enthought.traits.api import Any, Enum, Event, false, HasTraits, Instance, ReadOnly,\
                             RGBAColor, Str, Trait, Tuple


# Local relative imports
from base import bounds_to_coordinates, coordinates_to_bounds, intersect_coordinates, \
                 union_coordinates, does_disjoint_intersect_coordinates, \
                 disjoint_union_coordinates, transparent_color, coordinates_to_size, \
                 empty_rectangle, bounding_box, bounding_coordinates, add_rectangles, \
                 xy_in_bounds, BOTTOM_LEFT, TOP, BOTTOM, LEFT, RIGHT
from component import Component
from interactor import Interactor
from container import Container
from enable_traits import CURSOR_X, CURSOR_Y
from colors import ColorTrait

class AbstractWindow ( HasTraits ):

    # The top-level component that this window houses
    component     = Instance(Component)

    # A reference to the nested component that has focus.  This is part of the
    # manual mechanism for determining keyboard focus.
    focus_owner   = Instance(Interactor)

    # If set, this is the component to which all mouse events are passed,
    # bypassing the normal event propagation mechanism.
    mouse_owner   = Instance(Interactor)

    # The transform to apply to mouse event positions to put them into the
    # relative coordinates of the mouse_owner component.  Eventually this
    # should be a full transform; for now, it's just a tuple (dx,dy).
    mouse_owner_transform = Any
    
    bg_color      = ColorTrait("lightgray")
    window        = ReadOnly
    alt_pressed   = false
    ctrl_pressed  = false
    shift_pressed = false
    
    # A container that gets drawn after & on top of the main component, and
    # which receives events first.
    overlay = Instance(Container)
    
    # When the underlying toolkit control gets resized, this event gets set
    # to the new size of the window, expressed as a tuple (dx, dy).
    resized = Event
    
    # The previous component that handled an event.  Used to generate
    # mouse_enter and mouse_leave events.  Right now this can only be
    # None, self.component, or self.overlay.
    _prev_event_handler = Instance(Component)
    
    # (dx, dy) integer size of the Window.
    _size = Trait(None, Tuple)

    def __init__(self, **traits):
        self.window = self
        self._scroll_origin = (0.0, 0.0)
        self._update_region = None
        self._gc = None
        self._pointer_owner = None
        HasTraits.__init__(self, **traits)
       
        # Create a default component (if necessary):
        if self.component is None:
            self.component = Container()
        return

    def _component_changed(self, old, new):
        if old is not None:
            old.on_trait_change(self.component_bounds_changed, 'bounds', remove=True)
            old.window = None

        if new is None:
            self.component = Container()
            return
            
        new.window = self

        # If possible, size the new component according to the size of the 
        # toolkit control
        size = self._get_control_size()
        if (size is not None) and hasattr(self.component, "bounds"):
            new.on_trait_change(self.component_bounds_changed, 'bounds')
            if hasattr(self.component, "fit_window") and self.component.fit_window:
                self.component.outer_position = [0,0]
                self.component.outer_bounds = list(size)
            elif hasattr(self.component, "resizable"):
                if "h" in self.component.resizable:
                    self.component.outer_x = 0
                    self.component.outer_width = size[0]
                if "v" in self.component.resizable:
                    self.component.outer_y = 0
                    self.component.outer_height = size[1]
        self.redraw()
        return
    
    def component_bounds_changed(self, bounds):
        """
        Dynamic trait listener that handles our component changing its size;
        bounds is a length-2 list of [width, height].
        """
        pass

    def set_mouse_owner(self, mouse_owner, transform=None):
        "Handle the 'mouse_owner' being changed"
        if mouse_owner is None:
            self._release_mouse()
        else:
            self._capture_mouse()
        self.mouse_owner = mouse_owner
        self.mouse_owner_transform = transform
        return

    #---------------------------------------------------------------------------
    #  Generic mouse event handler:
    #---------------------------------------------------------------------------

    def _handle_mouse_event(self, event_name, event, cursor_check=0, set_focus=False):
        if self._size is None:
            # PZW: Hack!
            # We need to handle the cases when the window hasn't been painted yet, but
            # it's gotten a mouse event.  In such a case, we just ignore the mouse event.
            # If the window has been painted, then _size will have some sensible value.
            return
        
        mouse_event = self._create_mouse_event(event)
        mouse_owner = self.mouse_owner
        
        if mouse_owner is not None:
            # A mouse_owner has grabbed the mouse
            if self.mouse_owner_transform is not None:
                mouse_event.offset_xy(*self.mouse_owner_transform)
            mouse_owner.dispatch(mouse_event, event_name)
            self._pointer_owner = mouse_owner
        else:
            # Normal event handling loop
            if self.overlay is not None:
                # TODO: implement this...
                pass
            if (not mouse_event.handled) and (self.component is not None):
                # Test to see if we need to generate a mouse_leave event
                if self._prev_event_handler:
                    if not self._prev_event_handler.is_in(mouse_event.x, mouse_event.y):
                        self._prev_event_handler.dispatch(mouse_event, "mouse_leave")
                
                if self.component.is_in(mouse_event.x, mouse_event.y):
                    # Test to see if we need to generate a mouse_enter event
                    if self._prev_event_handler != self.component:
                        self._prev_event_handler = self.component
                        self.component.dispatch(mouse_event, "mouse_enter")
                    
                    # Fire the actual event
                    self.component.dispatch(mouse_event, event_name)
            

        # If this event requires setting the keyboard focus, set the first
        # component under the mouse pointer that accepts focus as the new focus
        # owner (otherwise, nobody owns the focus):
        if set_focus: 
            # If the mouse event was a click, then we set focus to ourselves
            if (self.component is not None) and (self.component.accepts_focus) and \
                    (mouse_event.left_down or mouse_event.middle_down or \
                    mouse_event.right_down or mouse_event.mouse_wheel != 0):
                new_focus_owner = self.component
                self._set_focus()
            else:
                new_focus_owner = None
                
            old_focus_owner  = self.focus_owner
            self.focus_owner = new_focus_owner
            if ((old_focus_owner is not None) and 
                (old_focus_owner is not new_focus_owner)):
                old_focus_owner.has_kdb_focus = False
            if new_focus_owner is not None:
                new_focus_owner.has_kbd_focus = True
                self._set_focus()
        return
    
    def set_tooltip(self, components):
        "Set the window's tooltip (if necessary)"
        if self.component:
            tooltip = self.component.tooltip
            if tooltip != self.tooltip:
                self.tooltip = tooltip
                self._set_tooltip(tooltip)
        return
    
    def redraw(self):
        """ Requests that the window be redrawn. """
        self._redraw()
        return
    
    def _needs_redraw(self, bounds):
        "Determine if a specified region intersects the update region"
        return does_disjoint_intersect_coordinates( self._update_region,
                                               bounds_to_coordinates( bounds ) )

    def _paint(self, event=None):
        size = self._get_control_size()
        if (self._size != tuple(size)) or (self._gc is None):
            self._gc = self._create_gc(size)
            self._size = tuple(size)
        gc = self._gc
        gc.clear(self.bg_color_)
        if hasattr(self.component, "do_layout"):
            self.component.do_layout()
        self.component.draw(gc, view_bounds=(0, 0, size[0], size[1]))
        self._window_paint(event)
        return
    
    def __getstate__(self):
        attribs = ("component", "bg_color", "overlay", "_scroll_origin")
        state = {}
        for attrib in attribs:
            state[attrib] = getattr(self, attrib)
        return state
    
    #---------------------------------------------------------------------------
    #  Abstract methods that must be implemented by concrete subclasses
    #---------------------------------------------------------------------------
    
    def set_drag_result(self, result):
        """ Sets the result that should be returned to the system from the
        handling of the current drag operation.  Valid result values are:
        "error", "none", "copy", "move", "link", "cancel".  These have the
        meanings associated with their WX equivalents.
        """
        raise NotImplementedError
    
    def _capture_mouse(self):
        "Capture all future mouse events"
        raise NotImplementedError        
    
    def _release_mouse(self):
        "Release the mouse capture"
        raise NotImplementedError        
    
    def _create_mouse_event(self, event):
        "Convert a GUI toolkit mouse event into a MouseEvent"
        raise NotImplementedError        
    
    def _redraw(self, coordinates=None):
        "Request a redraw of the window"
        raise NotImplementedError  
    
    def _get_control_size(self):
        "Get the size of the underlying toolkit control"
        raise NotImplementedError
    
    def _create_gc(self, size, pix_format = "bgr24"):
        "Create a Kiva graphics context of a specified size"
        raise NotImplementedError  
        
    def _window_paint(self, event):
        "Do a GUI toolkit specific screen update"
        raise NotImplementedError  
        
    def set_pointer(self, pointer):
        "Sets the current cursor shape"
        raise NotImplementedError  
        
    def set_tooltip(self, tooltip):
        "Sets the current tooltip for the window"
        raise NotImplementedError  
        
    def _set_timer_interval(self, component, interval):
        "Set up or cancel a timer for a specified component"
        raise NotImplementedError  

    def _set_focus(self):
        "Sets this window to have keyboard focus"
        raise NotImplementedError

    #---------------------------------------------------------------------------
    # Wire up the mouse event handlers
    #---------------------------------------------------------------------------
 
    def _on_left_down ( self, event ):
        self._handle_mouse_event( 'left_down', event, set_focus = True )
 
    def _on_left_up ( self, event ):
        self._handle_mouse_event( 'left_up', event )
 
    def _on_left_dclick ( self, event ):
        self._handle_mouse_event( 'left_dclick', event )
 
    def _on_right_down ( self, event ):
        self._handle_mouse_event( 'right_down', event, set_focus = True )
 
    def _on_right_up ( self, event ):
        self._handle_mouse_event( 'right_up', event )
 
    def _on_right_dclick ( self, event ):
        self._handle_mouse_event( 'right_dclick', event )
 
    def _on_middle_down ( self, event ):
        self._handle_mouse_event( 'middle_down', event )
 
    def _on_middle_up ( self, event ):
        self._handle_mouse_event( 'middle_up', event )
 
    def _on_middle_dclick ( self, event ):
        self._handle_mouse_event( 'middle_dclick', event )
 
    def _on_mouse_move ( self, event ):
        self._handle_mouse_event( 'mouse_move', event, 1 )
        
    def _on_mouse_wheel ( self, event ):
        self._handle_mouse_event( 'mouse_wheel', event )
 
    def _on_mouse_enter ( self, event ):
        self._handle_mouse_event( 'mouse_enter', event )
 
    def _on_mouse_leave ( self, event ):
        self._handle_mouse_event( 'mouse_leave', event, -1 )

    # Additional event handlers that are not part of normal Interactors
    def _on_window_enter(self, event):
        # TODO: implement this to generate a mouse_leave on self.component
        pass
    
    def _on_window_leave(self, event):
        if self._prev_event_handler:
            mouse_event = self._create_mouse_event(event)
            self._prev_event_handler.dispatch(mouse_event, "mouse_leave")
            self._prev_event_handler = None
        return


# EOF
