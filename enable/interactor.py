# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
""" Defines the Interactor class """

# Enthought library imports
from kiva.api import affine_identity
from traits.api import Any, Bool, HasTraits, List, Property, Str, Trait

# Local relative imports
from enable.colors import ColorTrait
from .enable_traits import cursor_style_trait, Pointer


class Interactor(HasTraits):
    """
    The base class of any Enable object that receives keyboard and mouse
    events.  Adds the notion of "state" which determines which set of
    event handler methods get called.  The default state is "normal", so
    a "left_down" event would be dispatched by calling::

        self.normal_left_down(event)

    The event suffixes are:

    - left_down
    - left_up
    - left_dclick
    - right_down
    - right_up
    - right_dclick
    - middle_down
    - middle_up
    - middle_dclick
    - mouse_move
    - mouse_wheel
    - mouse_enter
    - mouse_leave
    - key_pressed
    - key_released
    - character
    - dropped_on
    - drag_over
    - drag_enter
    - drag_leave
    """

    # Name of the object's event state.  Used as a prefix when looking up
    # which set of event handlers should be used for MouseEvents and KeyEvents.
    # Subclasses should override this with an enumeration of their possible
    # states.
    event_state = Str("normal")

    # The cursor shape that should be displayed when this interactor is
    # "active"
    pointer = Pointer

    # The "tooltip" to display if a user mouse-overs this interactor
    tooltip = Trait(None, None, Str)

    # The cursor "style" to use
    cursor_style = cursor_style_trait

    # The color of the cursor...
    # PZW: Figure out the right type for this..
    cursor_color = ColorTrait

    # Whether or not the interactor accepts keyboard focus
    accepts_focus = Bool(True)

    # The tools that are registered as listeners.
    tools = List

    # The tool that is currently active.
    active_tool = Property

    # If True, then marks events as "handled" if there is a handler function
    # defined.  This makes it easy to write simple components that respond
    # to events, but more complex tools will probably want this turned off.
    auto_handle_event = Bool(True)

    # Shadow trait for the **active_tool** property.  Must be an instance of
    # BaseTool or one of its subclasses.
    _active_tool = Any

    def dispatch(self, event, suffix):
        """ Public method for sending mouse/keyboard events to this interactor.
        Subclasses may override this to customize the public dispatch behavior.

        Parameters
        ==========
        event : enable.BaseEvent instance
            The event to dispach
        suffix : string
            The type of event that occurred.  See class docstring for the
            list of possible suffixes.
        """
        self._dispatch_stateful_event(event, suffix)

    def get_event_transform(self, event=None, suffix=""):
        """ Returns the 3x3 transformation matrix that this interactor will
        apply to the event (if any).
        """
        return affine_identity()

    def _dispatch_stateful_event(self, event, suffix):
        """
        Protected method to dispatch a mouse or keyboard based on the current
        event_state.  Subclasses can call this from within customized
        event handling logic in dispatch().
        """
        handler = getattr(self, self.event_state + "_" + suffix, None)
        if handler is not None:
            handler(event)
            if self.auto_handle_event:
                event.handled = True
