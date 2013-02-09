#------------------------------------------------------------------------------
#  Copyright (c) 2013, Enthought, Inc.
#  All rights reserved.
#------------------------------------------------------------------------------

# traits imports
from traits.api import Dict, Instance, List, Str

# local imports
from container import Container
from layout.layout_manager import LayoutManager

class ConstraintsContainer(Container):
    """ A Container which lays out its child components using a
    constraints-based layout solver.

    """

    # The ID for this component. This ID can be used by the layout constraints
    # when referencing the container.
    id = "parent"

    # The layout constraints for this container.
    layout_constraints = List

    # A dictionary of components added to this container
    _component_map = Dict

    # All the hard constraints for child components
    _hard_constraints_map = Dict(Str, List)
    _hard_constraints = List

    # The size constraints for child components
    _size_constraints_map = Dict(Str, List)
    _size_constraints = List

    # The casuarius solver
    _layout_manager = Instance(LayoutManager)

    #------------------------------------------------------------------------
    # Public methods
    #------------------------------------------------------------------------

    def relayout(self):
        """ Re-run the constraints solver in response to a resize or
        constraints modification.
        """
        pass

    #------------------------------------------------------------------------
    # Traits methods
    #------------------------------------------------------------------------

    def _layout_constraints_changed(self):
        """ Invalidate the layout when constraints change
        """
        self.relayout()

    def _layout_constraints_items_changed(self, event):
        """ Invalidate the layout when constraints change
        """
        self.relayout()

    def __component_map_default(self):
        """ The default component map should include this container so that
        name collisions are avoided.
        """
        return {self.id:None}

    def __components_items_changed(self, event):
        """ Make sure components that are added can be used with constraints.
        """
        # Check the added components
        self._check_and_add_components(event.added)

        # Remove stale components from the map
        for item in event.removed:
            key = item.id
            del self._hard_constraints_map[key]
            del self._size_constraints_map[key]
            del self._component_map[key]

        # Update the fixed constraints
        self._update_fixed_constraints()

    def __components_changed(self, new):
        """ Make sure components that are added can be used with constraints.
        """
        # Clear the component map
        self._component_map = self.__component_map_default()

        # Check the new components
        self._check_and_add_components(new)

        # Update the fixed constraints
        self._update_fixed_constraints()

    def __layout_manager_default(self):
        """ Create the layout manager.
        """
        lm = LayoutManager()
        lm.initialize([])
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
            elif item.id in self._component_map:
                msg = "A Component with id '{0}' has already been added."
                raise ValueError(msg.format(key))

            self._hard_constraints_map[key] = item.hard_constraints
            self._size_constraints_map[key] = item.size_constraints
            self._component_map[key] = item

    def _update_fixed_constraints(self):
        """ Resolve the differences between the list of constraints and the
        map of child component constraints for both types of fixed constraints.
        """
        old_cns, all_new_cns = [], []
        for name in ('hard', 'size'):
            map_attr = getattr(self, '_{0}_constraints_map'.format(name))
            list_name = '_{0}_constraints'.format(name)
            old_cns.extend(getattr(self, list_name))
            new_cns = []
            for item in map_attr.itervalues():
                new_cns.extend(item)
            all_new_cns.extend(new_cns)
            setattr(self, list_name, new_cns)

        self._layout_manager.replace_constraints(old_cns, all_new_cns)

