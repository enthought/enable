""" Defines the enable Canvas class """


# Enthought library imports
from enthought.traits.api import Trait, Tuple


# Local relative imports
from container import Container


class Canvas(Container):
    """
    An infinite canvas with components on it.  It can optionally be given
    a "view region" which will be used as the notional bounds of the
    canvas in all operations that require bounds.

    A Canvas can be nested inside another container, but usually a
    viewport is more appropriate.
    """
    

    # This optional tuple of (x,y,x2,y2) allows viewports to inform the canvas of
    # the "region of interest" that it should use when computing its notional
    # bounds for clipping and event handling purposes.  If this trait is None,
    # then the canvas really does behave as if it has no bounds.
    view_bounds = Trait(None, None, Tuple)


    #------------------------------------------------------------------------
    # Inherited traits
    #------------------------------------------------------------------------

    # Use the auto-size/fit_components mechanism to ensure that the bounding
    # box around our inner components gets updated properly.
    auto_size = True
    fit_components = "hv"

    # The following traits are ignored, but we set them to sensible values.
    fit_window = False
    resizable = "hv"

    #------------------------------------------------------------------------
    # Protected traits
    #------------------------------------------------------------------------

    # The (x, y, x2, y2) coordinates of the bounding box of the components
    _bounding_box = Tuple((0,0,100,100))

    def compact(self):
        """
        Wraps the superclass method to also take into account the view
        bounds (if they are present
        """
        self._bounding_box = self._calc_bounding_box()
        self._view_bounds_changed()

    def is_in(self, x, y):
        return True
    
    def add(self, *components):
        """ Adds components to this container """
        for component in components:
            if component.container is not None:
                component.container.remove(component)
            component.container = self
        self._components.extend(components)

        # Expand our bounds if necessary
        if self._should_compact():
            self.compact()

        self.invalidate_draw()
        return


    #------------------------------------------------------------------------
    # Protected methods
    #------------------------------------------------------------------------
        
    def _should_compact(self):
        if self.auto_size:
            if self.view_bounds is not None:
                llx, lly = self.view_bounds[:2]
            else:
                llx = lly = 0
            for component in self.components:
                if (component.outer_x2 >= self.width) or \
                   (component.outer_y2 >= self.height) or \
                   (component.outer_x < llx) or (component.outer_y < lly):
                    return True
        else:
            return False
            

    #------------------------------------------------------------------------
    # Event handlers
    #------------------------------------------------------------------------

    def _view_bounds_changed(self):
        llx, lly, urx, ury = self._bounding_box
        if self.view_bounds is not None:
            x, y, x2, y2 = self.view_bounds
            llx = min(llx, x)
            lly = min(lly, y)
            urx = max(urx, x2)
            ury = max(ury, y2)
        self.bounds = [urx - llx + 1, ury - lly + 1]

