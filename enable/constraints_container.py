#------------------------------------------------------------------------------
#  Copyright (c) 2013, Enthought, Inc.
#  All rights reserved.
#------------------------------------------------------------------------------

# traits imports
from traits.api import Dict

# local imports
from container import Container


class ConstraintsContainer(Container):
    """ A Container which lays out its child components using a
    constraints-based layout solver.

    """

    # A dictionary of components added to this container
    _component_map = Dict

    #------------------------------------------------------------------------
    # Public methods
    #------------------------------------------------------------------------

    def relayout(self):
        """ Re-run the constraints solver in response to a resize or
        component removal.
        """
        pass

    #------------------------------------------------------------------------
    # Traits methods
    #------------------------------------------------------------------------

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

