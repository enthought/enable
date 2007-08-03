""" The base class for moveable shapes. """


# Enthought library imports.
from enthought.enable2.api import ColorTrait, Component, Pointer
from enthought.traits.api import Float


class Shape(Component):
    """ The base class for moveable shapes. """

    #### 'Component' interface ################################################

    # The background color of this component.
    bgcolor = 'transparent'
    
    #### 'Shape' interface ####################################################

    # The fill color.
    fill_color = ColorTrait

    # The pointer for the 'normal' event state.
    normal_pointer = Pointer('arrow')

    # The pointer for the 'moving' event state.
    moving_pointer = Pointer('hand')

    #### 'Private' interface ##################################################

    # The difference between the location of a mouse-click and the component's
    # origin.
    _offset_x = Float
    _offset_y = Float
        
    ###########################################################################
    # 'Interactor' interface
    ###########################################################################

    def normal_key_pressed(self, event):
        """ Event handler. """
        
        print 'normal_key_pressed', event.character

        return
    
    def normal_left_down(self, event):
        """ Event handler. """

        self.event_state = 'moving'
        event.window.mouse_owner = self
        event.window.set_pointer(self.moving_pointer)

        self._offset_x = event.x - self.x
        self._offset_y = event.y - self.y

        self.container.bring_to_top(self)

        return

    def moving_mouse_move(self, event):
        """ Event handler. """

        self.position = [event.x - self._offset_x, event.y - self._offset_y]
        self.request_redraw()

        return

    def moving_left_up(self, event):
        """ Event handler. """

        self.event_state = 'normal'

        event.window.set_pointer(self.normal_pointer)
        event.window.mouse_owner = None
        
        self.request_redraw()

        return
    
    def moving_mouse_leave(self, event):
        """ Event handler. """

        self.moving_left_up(event)

        return

    ###########################################################################
    # Protected 'Shape' interface
    ###########################################################################

    def _get_fill_color(self, event_state):
        """ Return the fill color based on the event state. """

        if event_state == 'normal':
            fill_color = self.fill_color_

        else:
            r, g, b, a = self.fill_color_
            fill_color = (r, g, b, 0.5)

        return fill_color
    
#### EOF ######################################################################
