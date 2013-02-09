#------------------------------------------------------------------------------
#  Copyright (c) 2013, Enthought, Inc.
#  All rights reserved.
#------------------------------------------------------------------------------

# traits imports
from traits.api import Dict, Instance, List

# local imports
from container import Container
from layout.layout_managet import LayoutManager

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
    _hard_constraints = List

    # The size constraints for child components
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

    def __components_items_changed(self, event):
        """ Make sure components that are added can be used with constraints.
        """
        # Check the added components
        self._check_and_add_components(event.added)

        # Remove stale components from the map
        for item in event.removed:
            del self._component_map[item.id]

    def __components_changed(self, new):
        """ Make sure components that are added can be used with constraints.
        """
        # Clear the component map
        self._component_map = {}

        # Check the new components
        self._check_and_add_components(new)

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
            if len(item.id) == 0:
                msg = "Components added to a {0} must have a valid 'id' trait."
                name = type(self).__name__
                raise ValueError(msg.format(name))
            elif item.id in self._component_map:
                msg = "A Component with that id has already been added."
                raise ValueError(msg)

            self._component_map[item.id] = item

