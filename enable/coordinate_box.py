
# Enthought library imports
from traits.api import HasTraits, Property

# Local, relative imports
from enable_traits import bounds_trait, coordinate_trait


class CoordinateBox(HasTraits):
    """
    Represents a box in screen space, and provides convenience properties to
    access bounds and coordinates in a variety of ways.

    Primary attributes (not properties):
        position : [x, y]
        bounds : [width, height]

    Secondary attributes (properties):
        x, y   : coordinates of the lower-left pixel of the box
        x2, y2 : coordinates of the upper-right pixel of the box
        width  : the number of horizontal pixels in the box; equal to x2-x+1
        height : the number of vertical pixels in the box; equal to y2-y+1

    Note that setting x and y will modify the position, but setting any of the
    other secondary attributes will modify the bounds of the box.
    """

    bounds = bounds_trait

    # The position relative to the container.  If container is None, then
    # position will be set to (0,0).
    position = coordinate_trait

    x = Property

    y = Property

    x2 = Property

    y2 = Property

    width = Property

    height = Property

    def is_in(self, x, y):
        "Returns if the point x,y is in the box"
        p = self.position
        b = self.bounds
        dx = x - p[0]
        dy = y - p[1]
        return (dx >= 0) and (dx < b[0]) and (dy >= 0) and (dy < b[1])

    def as_coordinates(self):
        "Returns a 4-tuple (x, y, x2, y2)"
        p = self.position
        b = self.bounds
        return (p[0], p[1], p[0]+b[0]-1, p[1]+b[1]-1)

    #------------------------------------------------------------------------
    # Property setters and getters
    #------------------------------------------------------------------------

    def _get_x(self):
        return self.position[0]

    def _set_x(self, val):
        self.position[0] = val
        return

    def _get_y(self):
        return self.position[1]

    def _set_y(self, val):
        self.position[1] = val
        return

    def _get_width(self):
        return self.bounds[0]

    def _set_width(self, val):

        if isinstance(val, basestring):
            try:
                val = float(val)
            except:
                pass

        old_value = self.bounds[0]
        self.bounds[0] = val
        self.trait_property_changed( 'width', old_value, val )
        return

    def _get_height(self):
        return self.bounds[1]

    def _set_height(self, val):
        if isinstance(val, basestring):
            try:
                val = float(val)
            except:
                pass
        old_value = self.bounds[1]
        self.bounds[1] = val
        self.trait_property_changed( 'height', old_value, val )
        return

    def _get_x2(self):
        if self.bounds[0] == 0: return self.position[0]
        return self.position[0] + self.bounds[0] - 1

    def _set_x2(self, val):
        self.position[0] = val - self.bounds[0] + 1
        return

    def _old_set_x2(self, val):
        new_width = val - self.position[0] + 1
        if new_width < 0.0:
            raise RuntimeError, "Attempted to set negative component width."
        else:
            self.bounds[0] = new_width
        return

    def _get_y2(self):
        if self.bounds[1] == 0:
            return self.position[1]
        return self.position[1] + self.bounds[1] - 1

    def _set_y2(self, val):
        self.position[1] = val - self.bounds[1] + 1
        return

    def _old_set_y2(self, val):
        new_height = val - self.position[1] + 1
        if new_height < 0.0:
            raise RuntimeError, "Attempted to set negative component height."
        else:
            self.bounds[1] = new_height
        return


# EOF
