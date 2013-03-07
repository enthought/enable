""" A moveable box shape. """
from enable.primitives.shape import Shape


class Box(Shape):
    """ A moveable box shape. """

    # Protected 'Component' interface.
    #---------------------------------

    def _draw_mainlayer(self, gc, view_bounds=None, mode='default'):
        """ Draw the component. """
        with gc:
            gc.set_fill_color(self._get_fill_color(self.event_state))

            dx, dy = self.bounds
            x, y = self.position
            gc.rect(x, y, dx, dy)
            gc.fill_path()

            # Draw the shape's text.
            self._draw_text(gc)

        return
