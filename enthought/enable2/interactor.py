""" Defines the Interactor class """

# Enthought library imports
from enthought.traits.api import Any, Bool, HasTraits, List, Property, Str, Trait

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
        """ Dispatches a mouse event based on the current event state.
        
        Parameters
        ----------
        event : an Enable MouseEvent
            A mouse event.
        suffix : string
            The name of the mouse event as a suffix to the event state name,
            e.g. "_left_down" or "_window_enter".
        """

        # This hasattr check is necessary to ensure compatibility with Chaco
        # components.
        if not getattr(self, "use_draw_order", True):
            self._old_dispatch(event, suffix)
        else:
            self._new_dispatch(event, suffix)
        return


    def _dispatch_stateful_event(self, event, suffix):
        "Dispatches a MouseEvent based on the current event_state; private method."
        handler = getattr(self, self.event_state + "_" + suffix, None)
        if handler is not None:
            handler(event)
            if self.auto_handle_event:
                event.handled = True
        return

    
    def _new_dispatch(self, event, suffix):
        """ Dispatches a mouse event
        
        If the component has a **controller**, the method dispatches the event 
        to it, and returns. Otherwise, the following objects get a chance to 
        handle the event:
        
        1. The component's active tool, if any.
        2. Any overlays, in reverse order that they were added and are drawn.
        3. The component itself.
        4. Any underlays, in reverse order that they were added and are drawn.
        5. Any listener tools.
        
        If any object in this sequence handles the event, the method returns
        without proceeding any further through the sequence. If nothing
        handles the event, the method simply returns.
        """
        
        # Maintain compatibility with .controller for now
        if self.controller is not None:
            self.controller.dispatch(event, suffix)
            return
        
        if self._active_tool is not None:
            self._active_tool.dispatch(event, suffix)

        if event.handled:
            return
        
        # Dispatch to overlays in reverse of draw/added order
        for overlay in self.overlays[::-1]:
            overlay.dispatch(event, suffix)
            if event.handled:
                break
            
        if not event.handled:
            self._dispatch_stateful_event(event, suffix)
        
        if not event.handled:
            # Dispatch to underlays in reverse of draw/added order
            for underlay in self.underlays[::-1]:
                underlay.dispatch(event, suffix)
                if event.handled:
                    break
        
        # Now that everyone who might veto/handle the event has had a chance
        # to receive it, dispatch it to our list of listener tools.
        if not event.handled:
            for tool in self.tools:
                tool.dispatch(event, suffix)
        
        return

    def _old_dispatch(self, event, suffix):
        """ Dispatches a mouse event.
        
        If the component has a **controller**, the method dispatches the event 
        to it and returns. Otherwise, the following objects get a chance to 
        handle the event:
        
        1. The component's active tool, if any.
        2. Any listener tools.
        3. The component itself.
        
        If any object in this sequence handles the event, the method returns
        without proceeding any further through the sequence. If nothing
        handles the event, the method simply returns.
        
        """
        if self.controller is not None:
            self.controller.dispatch(event, suffix)
            return
        
        if self._active_tool is not None:
            self._active_tool.dispatch(event, suffix)

        if event.handled:
            return
        
        for tool in self.tools:
            tool.dispatch(event, suffix)
            if event.handled:
                return
        
        if not event.handled:
            self._dispatch_to_enable(event, suffix)
        return
# EOF
