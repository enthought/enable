""" Defines the Interactor class """

# Enthought library imports
from enthought.traits.api import false, HasTraits, Str, Trait, true

# Local relative imports
from enable_traits import bounds_trait, cursor_style_trait, Pointer
from enthought.enable2.traits.rgba_color_trait import RGBAColor

class Interactor(HasTraits):
    """
    The base class of any Enable object that receives keyboard and mouse
    events.  Adds the notion of "state" which determines which set of
    event handler methods get called.  The default state is "normal", so
    a "left_down" event would be dispatched by calling:
        self.normal_left_down(event)

    The event suffices are:

    left_down
    left_up
    left_dclick
    right_down
    right_up
    right_dclick
    middle_down
    middle_up
    middle_dclick
    mouse_move
    mouse_wheel
    mouse_enter
    mouse_leave
    key_pressed
    dropped_on
    drag_over
    drag_enter
    drag_leave
    """

    # Name of the object's event state.  Used as a prefix when looking up
    # which set of event handlers should be used for MouseEvents and KeyEvents.
    event_state = Str("normal")

    # The cursor shape that should be displayed when this interactor is "active"
    pointer = Pointer

    # The "tooltip" to display if a user mouse-overs this interactor
    tooltip = Trait(None, None, Str)

    # The cursor "style" to use
    cursor_style = cursor_style_trait

    # The color of the cursor...
    # PZW: Figure out the right type for this..
    cursor_color = RGBAColor

    # Whether or not the interactor accepts keyboard focus
    accepts_focus = true

    # If True, then marks events as "handled" if there is a handler function
    # defined.  This makes it easy to write simple components that respond
    # to events, but more complex tools will probably want this turned off.
    auto_handle_event = true

    def dispatch(self, event, suffix):
        "Public method to dispatch a MouseEvent based on the current event state"
        self._dispatch_stateful_event(event, suffix)
        return

    #------------------------------------------------------------------------
    # Mouse events
    #------------------------------------------------------------------------

    def _dispatch_stateful_event(self, event, suffix):
        "Dispatches a MouseEvent based on the current event_state; private method."
        handler = getattr(self, self.event_state + "_" + suffix, None)
        if handler is not None:
            handler(event)
            if self.auto_handle_event:
                event.handled = True
        return


# EOF
