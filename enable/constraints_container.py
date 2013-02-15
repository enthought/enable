#------------------------------------------------------------------------------
#  Copyright (c) 2013, Enthought, Inc.
#  All rights reserved.
#------------------------------------------------------------------------------

# traits imports
from traits.api import Callable, Dict, Either, Instance, List, Str

# local imports
from container import Container
from layout.layout_helpers import expand_constraints
from layout.layout_manager import LayoutManager


class ConstraintsContainer(Container):
    """ A Container which lays out its child components using a
    constraints-based layout solver.

    """

    # The ID for this component.
    id = "parent"

    # The layout constraints for this container.
    # This can either be a list or a callable. If it is a callable, it will be
    # called with a single argument, the ConstraintsContainer, and be expected
    # to return a list of constraints.
    layout_constraints = Either(List, Callable)

    # A copy of the layout constraints used when the constraints change
    _layout_constraints = List

    # A dictionary of components added to this container
    _component_map = Dict

    # All the hard constraints for child components
    _child_hard_constraints_map = Dict(Str, List)
    _child_hard_constraints = List

    # The size constraints for child components
    _child_size_constraints_map = Dict(Str, List)
    _child_size_constraints = List

    # The casuarius solver
    _layout_manager = Instance(LayoutManager)

    #------------------------------------------------------------------------
    # Public methods
    #------------------------------------------------------------------------

    def refresh_layout_constraints(self):
        """ Explicitly regenerate the container's constraints and refresh the
        layout.
        """
        self._layout_constraints_changed()

    def relayout(self):
        """ Re-run the constraints solver in response to a resize or
        constraints modification.
        """
        mgr_layout = self._layout_manager.layout
        constraints = self.constraints
        width_var = constraints.width
        height_var = constraints.height
        width, height = self.bounds
        def layout():
            for component in self._component_map.itervalues():
                cns = component.constraints
                component.position = (cns.left.value, cns.bottom.value)
                component.bounds = (cns.width.value, cns.height.value)
        mgr_layout(layout, width_var, height_var, (width, height))

        self.invalidate_draw()

    #------------------------------------------------------------------------
    # Traits methods
    #------------------------------------------------------------------------
    def _bounds_changed(self, old, new):
        """ Run the solver when the container's bounds change.
        """
        super(ConstraintsContainer, self)._bounds_changed(old, new)
        self.relayout()

    def _layout_constraints_changed(self):
        """ React to changes of the user controlled constraints.
        """
        if self.layout_constraints is None:
            return

        if callable(self.layout_constraints):
            new = self.layout_constraints(self)
        else:
            new = self.layout_constraints

        # Update the private constraints list. This will trigger the relayout.
        expand = expand_constraints
        constraints = self.constraints
        self._layout_constraints = [cns for cns in expand(constraints, new)]

    def __layout_constraints_changed(self, name, old, new):
        """ Invalidate the layout when the private constraints list changes.
        """
        self._layout_manager.replace_constraints(old, new)
        self.relayout()

    def __components_items_changed(self, event):
        """ Make sure components that are added can be used with constraints.
        """
        # Remove stale components from the map
        for item in event.removed:
            key = item.id
            del self._child_hard_constraints_map[key]
            del self._child_size_constraints_map[key]
            del self._component_map[key]

        # Check the added components
        self._check_and_add_components(event.added)

    def __components_changed(self, new):
        """ Make sure components that are added can be used with constraints.
        """
        # Clear the component maps
        self._component_map = {}
        self._child_hard_constraints_map = {}
        self._child_size_constraints_map = {}

        # Check the new components
        self._check_and_add_components(new)

    def __layout_manager_default(self):
        """ Create the layout manager.
        """
        lm = LayoutManager()

        constraints = self._hard_constraints + self._content_box_constraints()
        lm.initialize(constraints)
        return lm

    #------------------------------------------------------------------------
    # Protected methods
    #------------------------------------------------------------------------

    def _check_and_add_components(self, components):
        """ Make sure components can be used with constraints.
        """
        for item in components:
            key = item.id
            if len(key) == 0:
                msg = "Components added to a {0} must have a valid 'id' trait."
                name = type(self).__name__
                raise ValueError(msg.format(name))
            elif key in self._component_map:
                msg = "A Component with id '{0}' has already been added."
                raise ValueError(msg.format(key))
            elif key == self.id:
                msg = "Can't add a Component with the same id as its parent."
                raise ValueError(msg)

            self._child_hard_constraints_map[key] = item._hard_constraints
            self._child_size_constraints_map[key] = item._size_constraints
            self._component_map[key] = item

        # Update the fixed constraints
        self._update_fixed_constraints()

    def _content_box_constraints(self):
        """ Return the constraints which define the content box of this
        container.

        """
        cns = self.constraints
        contents_left = cns.contents_left
        contents_right = cns.contents_right
        contents_top = cns.contents_top
        contents_bottom = cns.contents_bottom

        # Add these to the namespace, but don't use them
        cns.contents_width = contents_right - contents_left
        cns.contents_height = contents_top - contents_bottom
        cns.contents_v_center = contents_bottom + cns.contents_height / 2.0
        cns.contents_h_center = contents_left + cns.contents_width / 2.0

        return [contents_left == cns.left + self.padding_left,
                contents_bottom == cns.bottom + self.padding_bottom,
                contents_right == cns.right - self.padding_right,
                contents_top == cns.top - self.padding_top,
            ]

    def _update_fixed_constraints(self):
        """ Resolve the differences between the list of constraints and the
        map of child component constraints for both types of fixed constraints.
        """
        old_cns, all_new_cns = [], []
        for name in ('hard', 'size'):
            map_attr = getattr(self, '_child_{0}_constraints_map'.format(name))
            list_name = '_child_{0}_constraints'.format(name)
            old_cns.extend(getattr(self, list_name))
            new_cns = []
            for item in map_attr.itervalues():
                new_cns.extend(item)
            all_new_cns.extend(new_cns)
            setattr(self, list_name, new_cns)

        self._layout_manager.replace_constraints(old_cns, all_new_cns)
        # Possibly regenerate the user-specified constraints
        self.refresh_layout_constraints()
