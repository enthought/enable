
from uuid import uuid4

# Enthought library imports
from traits.api import HasTraits, Enum, Instance, Property

# Local, relative imports
from enable_traits import bounds_trait, coordinate_trait
from layout.constraints_namespace import ConstraintsNamespace


ConstraintPolicyEnum = Enum('ignore', 'weak', 'medium', 'strong', 'required')


def add_symbolic_constraints(namespace):
    """ Add constraints to a namespace that are LinearExpressions of basic
    constraints.

    """
    bottom = namespace.bottom
    left = namespace.left
    width = namespace.width
    height = namespace.height

    namespace.right = left + width
    namespace.top = bottom + height
    namespace.h_center = left + width / 2.0
    namespace.v_center = bottom + height / 2.0


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

    #------------------------------------------------------------------------
    # Constraints-based layout
    #------------------------------------------------------------------------

    # A namespace containing the constraints for this CoordinateBox
    constraints = Instance(ConstraintsNamespace)

    # How strongly a layout box hugs it's width hint.
    hug_width = ConstraintPolicyEnum('strong')

    # How strongly a layout box hugs it's height hint.
    hug_height = ConstraintPolicyEnum('strong')

    # How strongly a layout box resists clipping its contents.
    resist_width = ConstraintPolicyEnum('strong')

    # How strongly a layout box resists clipping its contents.
    resist_height = ConstraintPolicyEnum('strong')

    # The list of hard constraints which must be applied to the object.
    _hard_constraints = Property

    # The list of size constraints to apply to the object.
    _size_constraints = Property

    #------------------------------------------------------------------------
    # Public methods
    #------------------------------------------------------------------------

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

    def _constraints_default(self):
        cns_names = ConstraintsNamespace(type(self).__name__, uuid4().hex)
        add_symbolic_constraints(cns_names)
        return cns_names

    def _get__hard_constraints(self):
        """ Generate the constraints which must always be applied.
        """
        constraints = self.constraints
        left = constraints.left
        bottom = constraints.bottom
        width = constraints.width
        height = constraints.height
        cns = [left >= 0, bottom >= 0, width >= 0, height >= 0]
        return cns

    def _get__size_constraints(self):
        """ Creates the list of size hint constraints for this box.
        """
        cns = []
        push = cns.append
        width_hint, height_hint = self.bounds
        constraints = self.constraints
        width = constraints.width
        height = constraints.height
        hug_width, hug_height = self.hug_width, self.hug_height
        resist_width, resist_height = self.resist_width, self.resist_height
        if width_hint >= 0:
            if hug_width != 'ignore':
                cn = (width == width_hint) | hug_width
                push(cn)
            if resist_width != 'ignore':
                cn = (width >= width_hint) | resist_width
                push(cn)
        if height_hint >= 0:
            if hug_height != 'ignore':
                cn = (height == height_hint) | hug_height
                push(cn)
            if resist_height != 'ignore':
                cn = (height >= height_hint) | resist_height
                push(cn)

        return cns


# EOF
