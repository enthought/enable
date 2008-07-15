""" A moveable circle shape. """


# Enthought library imports.
from enthought.traits.api import Float

# Local imports.
from shape import Shape


class Circle(Shape):
    """ A moveable circle shape. """

    #### 'Circle' interface ###################################################

    # The radius of the circle.
    radius = Float

    ###########################################################################
    # 'CoordinateBox' interface.
    ###########################################################################

    def _bounds_changed(self):
        """ Static trait change handler. """
        
        w, h = self.bounds

        radius = min(w, h) / 2.0

        return

    ###########################################################################
    # 'Component' interface.
    ###########################################################################

    def is_in(self, x, y):
        """ Return True if a point is considered to be 'in' the component. """

        return self._distance_between(self.centre, (x, y)) <= self.radius

    ###########################################################################
    # Protected 'Component' interface.
    ###########################################################################

    def _draw_mainlayer(self, gc, view_bounds=None, mode='default'):
        """ Draw the component. """
        
        gc.save_state()

        gc.set_fill_color(self._get_fill_color(self.event_state))
        
        x, y = self.position
        gc.arc(x + self.radius, y + self.radius, self.radius, 0, 2*3.14159, False)
        gc.fill_path()

        # Draw the shape's text.
        self._draw_text(gc)
        
        gc.restore_state()

        return

    ###########################################################################
    # 'Circle' interface.
    ###########################################################################

    def _radius_changed(self):
        """ Static trait change handler. """
        
        diameter = self.radius * 2

        self.bounds = [diameter, diameter]

        return
    
#### EOF ######################################################################
