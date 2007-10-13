"""
Define the event objects and traits used by Enable components.

For a list of all the possible event suffixes, see interactor.py.
"""

# Major library imports
from numpy import array, dot, eye, matrix, float64 

# Enthought imports
from enthought.kiva import affine
from enthought.traits.api import Any, Bool, Float, HasTraits, Int, Event, \
        List, ReadOnly


class BasicEvent(HasTraits):
    
    x = Float
    y = Float
    
    # True if the event has been handled.
    handled = Bool(False)

    # The AbstractWindow instance through/from which this event was fired.
    # Can be None.
    window = Any

    # (x,y) position stack; initialized to an empty list
    _pos_stack = List( () )
    
    # Affine transform stack; initialized to an empty list
    _transform_stack = List( () )

    # This is a list of objects that have transformed the event's
    # coordinates.  This can be used to recreate the dispatch path
    # that the event took.
    dispatch_history = List()

    def push_transform(self, transform, caller=None):
        """
        Saves the current transform in a stack and sets the given transform
        to be the active one.
        """
        x, y = dot(array((self.x, self.y, 1)), transform)[:2]
        self._pos_stack.append((self.x, self.y))
        self._transform_stack.append(transform)
        self.x = x
        self.y = y
        if caller is not None:
            self.dispatch_history.append(caller)
        return
    
    def pop(self, count=1, caller=None):
        """
        Restores a previous position of the event.  If **count** is provided,
        then pops **count** elements off of the event stack.
        """
        for i in range(count-1):
            self._pos_stack.pop()
            self._transform_stack.pop()
        self.x, self.y = self._pos_stack.pop()
        self._transform_stack.pop()
        if caller is not None:
            if caller == self.dispatch_history[-1]:
                self.dispatch_history.pop()
        return
    
    def offset_xy(self, origin_x, origin_y, caller=None):
        """
        Shifts this event to be in the coordinate frame whose origin, specified
        in the event's coordinate frame, is (origin_x, origin_y).
        
        Basically, a component calls event.offset_xy(*self.position) to shift
        the event into its own coordinate frame.
        """
        self.push_transform(affine.affine_from_translation(-origin_x, -origin_y))
        if caller is not None:
            self.dispatch_history.append(caller)
        return

    def scale_xy(self, scale_x, scale_y, caller=None):
        """
        Scales the event to be in the scale specified.
    
        A component calls event.scale_xy(scale) to scale the event into its own
        coordinate frame when the ctm has been scaled.  This operation is used
        for zooming.
        """
        # Note that the meaning of scale_x and scale_y for Enable
        # is the inverted from the meaning for Kiva.affine. 
        # TODO: Fix this discrepancy.
        self.push_transform(affine.affine_from_scale(1/scale_x, 1/scale_y))
        if caller is not None:
            self.dispatch_history.append(caller)
        return
        
    def net_transform(self):
        """
        Returns a single transformation (currently only (dx,dy)) that reflects
        the total amount of change from the original coordinates to the current
        offset coordinates stored in self.x and self.y.
        """
        transform = affine.affine_identity()
        for m in self._transform_stack:
            transform = dot(m, transform)
        return transform


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


class BlobEvent(BasicEvent):

    bid = Int(-1)
    
    v_x = Float(0.0)
    v_y = Float(0.0)





# EOF
