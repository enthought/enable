"""
Define the event objects and traits used by Enable components.

For a list of all the possible event suffixes, see interactor.py.
"""
    
from enthought.traits.api import Any, false, Float, HasTraits, Event, List, ReadOnly


class BasicEvent(HasTraits):
    
    x = Float
    y = Float
    
    # True if the event has been handled.
    handled = false

    # The AbstractWindow instance through/from which this event was fired.
    # Can be None.
    window = Any

    # (x,y) position stack; initialized to an empty list
    _pos_stack = List( () )
    
    def push_xy(self, x, y):
        """
        Saves the current position in a stack and sets self.x and self.y to
        a new value.
        """
        self._pos_stack.append((self.x, self.y))
        self.x = x
        self.y = y
        return
    
    def pop(self):
        "Restores the previous position of the event."
        self.x, self.y = self._pos_stack.pop()
        return
    
    def offset_xy(self, origin_x, origin_y):
        """
        Shifts this event to be in the coordinate frame whose origin, specified
        in the event's coordinate frame, is (origin_x, origin_y).
        
        Basically, a component calls event.offset_xy(*self.position) to shift
        the event into its own coordinate frame.
        """
        self.push_xy(self.x - origin_x, self.y - origin_y)
        return

    def net_transform(self):
        """
        Returns a single transformation (currently only (dx,dy)) that reflects
        the total amount of change from the original coordinates to the current
        offset coordinates stored in self.x and self.y.
        """
        pos = self._pos_stack
        if len(pos) == 0:
            return (0,0)
        else:
            original_x, original_y = pos[-1]
            return (original_x - self.x, original_y - self.y)
        
        
    
    #~ def xy(self, component):
        #~ "Return the mouse coordinates relative to a specified component"
        #~ cx, cy = component.location()
        #~ return(self.x - cx, self.y - cy)



class MouseEvent(BasicEvent):
    alt_down     = ReadOnly
    control_down = ReadOnly
    shift_down   = ReadOnly
    left_down    = ReadOnly
    middle_down  = ReadOnly
    right_down   = ReadOnly
    mouse_wheel  = ReadOnly
    
mouse_event_trait = Event(MouseEvent)


class DragEvent(BasicEvent):
    x0   = Float
    y0   = Float
    copy = ReadOnly
    obj  = ReadOnly 
    start_event = ReadOnly

drag_event_trait = Event(DragEvent)


class KeyEvent(BasicEvent):
    # 'character' is a single ASCII character or is a string describing the
    # high-bit and control characters.  (See subpackage wx.window.key_map)
    character    = ReadOnly 
    alt_down     = ReadOnly
    control_down = ReadOnly
    shift_down   = ReadOnly
    event        = ReadOnly    # what is this??

key_event_trait = Event( KeyEvent )

# EOF
