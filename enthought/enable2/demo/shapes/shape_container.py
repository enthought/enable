""" A shape container. """


# Enthought library imports.
from enthought.enable2.api import Container


class ShapeContainer(Container):
    """ The base class for moveable shapes. """
        
    ###########################################################################
    # 'ShapeContainer' interface
    ###########################################################################

    def bring_to_top(self, component):
        """ Bring a component to the top of the z-order.

        fixme: This seems like a horrible hack - is there a nice enable-y way?

        """

        components = self._components
        
        # Make sure that the component is the last in the list (if it is the
        # *only* one in the list then do nothing).
        if len(components) > 1:
            if components[-1] is not component:
                # After the append the list will contain two references to the 
                # component, one at its original location and one at the end...
                components.append(component)

                # ... and this removes the first one!
                components.remove(component)

        return
    
#### EOF ######################################################################
