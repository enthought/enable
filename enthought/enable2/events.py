"""
Define the event objects and traits used by Enable components.

For a list of all the possible event suffixes, see interactor.py.
"""
from numpy import matrix, float64 
from enthought.traits.api import Any, false, Float, HasTraits, Int, Event, List, ReadOnly


class BasicEvent(HasTraits):
    
    x = Float
    y = Float
    
    # The current transformation matrix
    cur_transform = Any 
    
    # True if the event has been handled.
    handled = false

    # The AbstractWindow instance through/from which this event was fired.
    # Can be None.
    window = Any

    # (x,y) position stack; initialized to an empty list
    _pos_stack = List( () )
    
    # Affine transform stack; initialized to an empty list
    _transform_stack = List( () )

    
    def push_transform(self, transform):
        """
        Saves the current transform in a stack and sets the given transform
        to be the active one.
        """
        xy_matrix = self.make_xy_matrix(self.x, self.y)
        transformed = transform * xy_matrix
        x, y = self.get_xy_position(transformed)
        self._pos_stack.append((self.x, self.y))
        self._transform_stack.append(transform)
        self.x = x
        self.y = y
        self.cur_transform = transform
        return
    
    def pop(self, count=1):
        """
        Restores a previous position of the event.  If **count** is provided,
        then pops **count** elements off of the event stack.
        """
        for i in range(count-1):
            self._pos_stack.pop()
            self._transform_stack.pop()
        self.x, self.y = self._pos_stack.pop()
        self.cur_transform = self._transform_stack.pop()
        return
    
    def offset_xy(self, origin_x, origin_y):
        """
        Shifts this event to be in the coordinate frame whose origin, specified
        in the event's coordinate frame, is (origin_x, origin_y).
        
        Basically, a component calls event.offset_xy(*self.position) to shift
        the event into its own coordinate frame.
        """
        translation_matrix = self._translate(-origin_x, -origin_y)
        self.push_transform(translation_matrix)
        return

    def scale_xy(self, scale_x, scale_y):
        """
        Scales the event to be in the scale specified.
    
        A component calls event.scale_xy(scale) to scale the event
        into its own coordinate frame when the ctm has been scaled.  This operation
        is used for zooming.
        """
        scale_matrix = self._scale(scale_x, scale_y)
        self.push_transform(scale_matrix)
        return
        
    def net_transform(self):
        """
        Returns a single transformation (currently only (dx,dy)) that reflects
        the total amount of change from the original coordinates to the current
        offset coordinates stored in self.x and self.y.
        """
        pos = self._pos_stack
        if len(pos) == 0:
            return self.identity_transform()
        else:
            return self.current_transform()

    def current_transform(self):
        """
        Returns a copy of the current transformation matrix.
        """
        transform = self.identity_transform()
        for m in self._transform_stack:
            transform = m * transform
        return transform

    def get_xy_position(self, transform):
        return (transform[0,0], transform[1,0])

    def make_xy_matrix(self, x, y):
        return matrix( [[x], [y], [1]], float64)

    def identity_transform(self):
        """
        Returns a new identity affine_transform
        """
        return matrix( [[1,0,0],[0,1,0],[0,0,1]], float64)

    def _scale( self, sx, sy ):
        """ Returns the matrix that must be multiplied in order to scale.
        """
        scaled = self.identity_transform()
        scaled[0] /= sx
        scaled[1] /= sy
        return scaled

    def _rotate( self, angle ):
        """ Returns the matrix that must be multiplied in order to rotate
            by angle in radians.
        """
        pass

    def _translate( self, x, y ):
        """ Returns the matrix that must be multiplied to translate 
            by (x,y).  
        """
        r = self.identity_transform()
        r[0,2] = x
        r[1,2] = y
        return r

    
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


class BlobEvent(BasicEvent):

    bid = Int(-1)
    
    v_x = Float(0.0)
    v_y = Float(0.0)





# EOF
