# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
# Enthought library imports
from traits.api import HasTraits, Enum, Instance, Property, Tuple

# Local, relative imports
from .enable_traits import bounds_trait, coordinate_trait

try:
    import kiwisolver
except ImportError:
    ENABLE_CONSTRAINTS = False
else:
    ENABLE_CONSTRAINTS = True
    from .layout.constraints_namespace import ConstraintsNamespace
    from .layout.ab_constrainable import ABConstrainable
    from .layout.utils import (
        add_symbolic_constraints,
        STRENGTHS,
        get_from_constraints_namespace,
    )

    ConstraintPolicyEnum = Enum("ignore", *STRENGTHS)

    del kiwisolver


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

    # ------------------------------------------------------------------------
    # Constraints-based layout
    # ------------------------------------------------------------------------

    if ENABLE_CONSTRAINTS:

        # A read-only symbolic object that represents the left boundary of
        # the component
        left = Property(fget=get_from_constraints_namespace)

        # A read-only symbolic object that represents the right boundary
        # of the component
        right = Property(fget=get_from_constraints_namespace)

        # A read-only symbolic object that represents the bottom boundary
        # of the component
        bottom = Property(fget=get_from_constraints_namespace)

        # A read-only symbolic object that represents the top boundary of
        # the component
        top = Property(fget=get_from_constraints_namespace)

        # A read-only symbolic object that represents the width of the
        # component
        layout_width = Property(fget=get_from_constraints_namespace)

        # A read-only symbolic object that represents the height of the
        # component
        layout_height = Property(fget=get_from_constraints_namespace)

        # A read-only symbolic object that represents the vertical center
        # of the component
        v_center = Property(fget=get_from_constraints_namespace)

        # A read-only symbolic object that represents the horizontal
        # center of the component
        h_center = Property(fget=get_from_constraints_namespace)

        # A size hint for the layout
        layout_size_hint = Tuple(0.0, 0.0)

        # How strongly a layout box hugs it's width hint.
        hug_width = ConstraintPolicyEnum("weak")

        # How strongly a layout box hugs it's height hint.
        hug_height = ConstraintPolicyEnum("weak")

        # How strongly a layout box resists clipping its contents.
        resist_width = ConstraintPolicyEnum("strong")

        # How strongly a layout box resists clipping its contents.
        resist_height = ConstraintPolicyEnum("strong")

        # A namespace containing the constraints for this CoordinateBox
        _constraints_vars = Instance(ConstraintsNamespace)

        # The list of hard constraints which must be applied to the object.
        _hard_constraints = Property

        # The list of size constraints to apply to the object.
        _size_constraints = Property

    # ------------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------------

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
        return (p[0], p[1], p[0] + b[0] - 1, p[1] + b[1] - 1)

    # ------------------------------------------------------------------------
    # Property setters and getters
    # ------------------------------------------------------------------------

    def _get_x(self):
        return self.position[0]

    def _set_x(self, val):
        self.position[0] = val

    def _get_y(self):
        return self.position[1]

    def _set_y(self, val):
        self.position[1] = val

    def _get_width(self):
        return self.bounds[0]

    def _set_width(self, val):

        if isinstance(val, str):
            try:
                val = float(val)
            except ValueError:
                pass

        old_value = self.bounds[0]
        self.bounds[0] = val
        self.trait_property_changed("width", old_value, val)

    def _get_height(self):
        return self.bounds[1]

    def _set_height(self, val):
        if isinstance(val, str):
            try:
                val = float(val)
            except ValueError:
                pass
        old_value = self.bounds[1]
        self.bounds[1] = val
        self.trait_property_changed("height", old_value, val)

    def _get_x2(self):
        if self.bounds[0] == 0:
            return self.position[0]
        return self.position[0] + self.bounds[0] - 1

    def _set_x2(self, val):
        self.position[0] = val - self.bounds[0] + 1

    def _old_set_x2(self, val):
        new_width = val - self.position[0] + 1
        if new_width < 0.0:
            raise RuntimeError("Attempted to set negative component width.")
        else:
            self.bounds[0] = new_width

    def _get_y2(self):
        if self.bounds[1] == 0:
            return self.position[1]
        return self.position[1] + self.bounds[1] - 1

    def _set_y2(self, val):
        self.position[1] = val - self.bounds[1] + 1

    def _old_set_y2(self, val):
        new_height = val - self.position[1] + 1
        if new_height < 0.0:
            raise RuntimeError("Attempted to set negative component height.")
        else:
            self.bounds[1] = new_height

    if ENABLE_CONSTRAINTS:

        def __constraints_vars_default(self):
            obj_name = self.id if hasattr(self, "id") else ""
            cns_names = ConstraintsNamespace(type(self).__name__, obj_name)
            add_symbolic_constraints(cns_names)
            return cns_names

        def _get__hard_constraints(self):
            """ Generate the constraints which must always be applied.
            """
            left = self.left
            bottom = self.bottom
            width = self.layout_width
            height = self.layout_height
            cns = [left >= 0, bottom >= 0, width >= 0, height >= 0]
            return cns

        def _get__size_constraints(self):
            """ Creates the list of size hint constraints for this box.
            """
            cns = []
            push = cns.append
            width_hint, height_hint = self.layout_size_hint
            width = self.layout_width
            height = self.layout_height
            hug_width, hug_height = self.hug_width, self.hug_height
            resist_width, resist_height = self.resist_width, self.resist_height
            if width_hint >= 0:
                if hug_width != "ignore":
                    cn = (width == width_hint) | hug_width
                    push(cn)
                if resist_width != "ignore":
                    cn = (width >= width_hint) | resist_width
                    push(cn)
            if height_hint >= 0:
                if hug_height != "ignore":
                    cn = (height == height_hint) | hug_height
                    push(cn)
                if resist_height != "ignore":
                    cn = (height >= height_hint) | resist_height
                    push(cn)

            return cns


if ENABLE_CONSTRAINTS:
    # Register with ABConstrainable so that layout helpers will recognize
    # CoordinateBox instances.
    ABConstrainable.register(CoordinateBox)
