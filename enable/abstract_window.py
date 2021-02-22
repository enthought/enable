# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
from functools import reduce

# Major library imports
from numpy import dot

# Enthought library imports
from traits.api import (
    Any, Bool, Event, Float, HasTraits, Instance, List, Property, Trait, Tuple,
)


# Local relative imports
from .base import (
    bounds_to_coordinates, does_disjoint_intersect_coordinates, union_bounds,
)
from .colors import ColorTrait
from .component import Component
from .container import Container
from .interactor import Interactor


def Alias(name):
    return Property(
        lambda obj: getattr(obj, name),
        lambda obj, val: setattr(obj, name, val),
    )


class AbstractWindow(HasTraits):

    # The top-level component that this window houses
    component = Instance(Component)

    # A reference to the nested component that has focus.  This is part of the
    # manual mechanism for determining keyboard focus.
    focus_owner = Instance(Interactor)

    # If set, this is the component to which all mouse events are passed,
    # bypassing the normal event propagation mechanism.
    mouse_owner = Instance(Interactor)

    # The transform to apply to mouse event positions to put them into the
    # relative coordinates of the mouse_owner component.
    mouse_owner_transform = Any()

    # When a component captures the mouse, it can optionally store a
    # dispatch order for events (until it releases the mouse).
    mouse_owner_dispatch_history = Trait(None, None, List)

    # A scaling constant applied to any GraphicsContext used for drawing the
    # window's component.
    base_pixel_scale = Float(1.0)

    # When True, allow `base_pixel_scale` to be greater than 1 if the
    # underlying toolkit supports it.
    high_resolution = Bool(True)

    # The background window of the window.  The entire window first gets
    # painted with this color before the component gets to draw.
    bgcolor = ColorTrait("sys_window")

    # Unfortunately, for a while, there was a naming inconsistency and the
    # background color trait named "bg_color".  This is still provided for
    # backwards compatibility but should not be used in new code.
    bg_color = Alias("bgcolor")

    alt_pressed = Bool(False)
    ctrl_pressed = Bool(False)
    shift_pressed = Bool(False)

    # A container that gets drawn after & on top of the main component, and
    # which receives events first.
    overlay = Instance(Container)

    # When the underlying toolkit control gets resized, this event gets set
    # to the new size of the window, expressed as a tuple (dx, dy).
    resized = Event

    # Whether to enable damaged region handling
    use_damaged_region = Bool(False)

    # The previous component that handled an event.  Used to generate
    # mouse_enter and mouse_leave events.  Right now this can only be
    # None, self.component, or self.overlay.
    _prev_event_handler = Instance(Component)

    # (dx, dy) integer size of the Window.
    _size = Trait(None, Tuple)

    # The regions to update upon redraw
    _update_region = Any

    # When exceeding this, the entire window is marked damaged to save memory
    MAX_DAMAGED_REGIONS = 100

    # -------------------------------------------------------------------------
    #  Abstract methods that must be implemented by concrete subclasses
    # -------------------------------------------------------------------------

    def set_drag_result(self, result):
        """ Sets the result that should be returned to the system from the
        handling of the current drag operation.  Valid result values are:
        "error", "none", "copy", "move", "link", "cancel".  These have the
        meanings associated with their WX equivalents.
        """
        raise NotImplementedError

    def _capture_mouse(self):
        "Capture all future mouse events"
        raise NotImplementedError

    def _release_mouse(self):
        "Release the mouse capture"
        raise NotImplementedError

    def _create_key_event(self, event):
        "Convert a GUI toolkit key event into a KeyEvent"
        raise NotImplementedError

    def _create_mouse_event(self, event):
        "Convert a GUI toolkit mouse event into a MouseEvent"
        raise NotImplementedError

    def _redraw(self, coordinates=None):
        """ Request a redraw of the window, within just the (x,y,w,h)
        coordinates (if provided), or over the entire window if coordinates is
        None.
        """
        raise NotImplementedError

    def _get_control_size(self):
        "Get the size of the underlying toolkit control"
        raise NotImplementedError

    def _create_gc(self, size, pix_format="bgr24"):
        """ Create a Kiva graphics context of a specified size.  This method
        only gets called when the size of the window itself has changed.  To
        perform pre-draw initialization every time in the paint loop, use
        _init_gc().
        """
        raise NotImplementedError

    def _init_gc(self):
        """ Gives a GC a chance to initialize itself before components perform
        layout and draw.  This is called every time through the paint loop.
        """
        gc = self._gc
        if self._update_region == [] or not self.use_damaged_region:
            self._update_region = None
        if self._update_region is None:
            gc.clear(self.bgcolor_)
        else:
            # Fixme: should use clip_to_rects
            update_union = reduce(union_bounds, self._update_region)
            gc.clip_to_rect(*update_union)

    def _window_paint(self, event):
        "Do a GUI toolkit specific screen update"
        raise NotImplementedError

    def set_pointer(self, pointer):
        "Sets the current cursor shape"
        raise NotImplementedError

    def set_timer_interval(self, component, interval):
        "Set up or cancel a timer for a specified component"
        raise NotImplementedError

    def _set_focus(self):
        "Sets this window to have keyboard focus"
        raise NotImplementedError

    def screen_to_window(self, x, y):
        "Returns local window coordinates for given global screen coordinates"
        raise NotImplementedError

    def get_pointer_position(self):
        "Returns the current pointer position in local window coordinates"
        raise NotImplementedError

    # ------------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------------

    def __init__(self, **traits):
        self._scroll_origin = (0.0, 0.0)
        self._update_region = None
        self._gc = None
        self._pointer_owner = None
        HasTraits.__init__(self, **traits)

        # Create a default component (if necessary):
        if self.component is None:
            self.component = Container()

    def _component_changed(self, old, new):
        if old is not None:
            old.on_trait_change(
                self.component_bounds_changed, "bounds", remove=True
            )
            old.window = None

        if new is None:
            self.component = Container()
            return

        new.window = self

        # If possible, size the new component according to the size of the
        # toolkit control
        size = self._get_control_size()
        if (size is not None) and hasattr(self.component, "bounds"):
            new.on_trait_change(self.component_bounds_changed, "bounds")
            if getattr(self.component, "fit_window", False):
                self.component.outer_position = [0, 0]
                self.component.outer_bounds = list(size)
            elif hasattr(self.component, "resizable"):
                if "h" in self.component.resizable:
                    self.component.outer_x = 0
                    self.component.outer_width = size[0]
                if "v" in self.component.resizable:
                    self.component.outer_y = 0
                    self.component.outer_height = size[1]
        self._update_region = None
        self.redraw()

    def component_bounds_changed(self, bounds):
        """
        Dynamic trait listener that handles our component changing its size;
        bounds is a length-2 list of [width, height].
        """
        self.invalidate_draw()
        pass

    def set_mouse_owner(self, mouse_owner, transform=None, history=None):
        "Handle the 'mouse_owner' being changed"
        if mouse_owner is None:
            self._release_mouse()
            self.mouse_owner = None
            self.mouse_owner_transform = None
            self.mouse_owner_dispatch_history = None
        else:
            self._capture_mouse()
            self.mouse_owner = mouse_owner
            self.mouse_owner_transform = transform
            self.mouse_owner_dispatch_history = history

    def invalidate_draw(self, damaged_regions=None, self_relative=False):
        if (damaged_regions is not None
                and self._update_region is not None
                and len(self._update_region) < self.MAX_DAMAGED_REGIONS):
            self._update_region += damaged_regions
        else:
            self._update_region = None

    # -------------------------------------------------------------------------
    #  Generic keyboard event handler:
    # -------------------------------------------------------------------------
    def _handle_key_event(self, event_type, event):
        """ **event** should be a toolkit-specific opaque object that will
        be passed in to the backend's _create_key_event() method. It can
        be None if the the toolkit lacks a native "key event" object.

        Returns True if the event has been handled within the Enable object
        hierarchy, or False otherwise.
        """
        # Generate the Enable event
        key_event = self._create_key_event(event_type, event)
        if key_event is None:
            return False

        self.shift_pressed = key_event.shift_down
        self.alt_pressed = key_event.alt_down
        self.control_pressed = key_event.control_down

        # Dispatch the event to the correct component
        mouse_owner = self.mouse_owner
        if mouse_owner is not None:
            history = self.mouse_owner_dispatch_history
            if history is not None and len(history) > 0:
                # Assemble all the transforms
                transforms = [c.get_event_transform() for c in history]
                total_transform = reduce(dot, transforms[::-1])
                key_event.push_transform(total_transform)
            elif self.mouse_owner_transform is not None:
                key_event.push_transform(self.mouse_owner_transform)

            mouse_owner.dispatch(key_event, event_type)
        else:
            # Normal event handling loop
            if (not key_event.handled) and (self.component is not None):
                if self.component.is_in(key_event.x, key_event.y):
                    # Fire the actual event
                    self.component.dispatch(key_event, event_type)

        return key_event.handled

    # -------------------------------------------------------------------------
    #  Generic mouse event handler:
    # -------------------------------------------------------------------------
    def _handle_mouse_event(self, event_name, event, set_focus=False):
        """ **event** should be a toolkit-specific opaque object that will
        be passed in to the backend's _create_mouse_event() method.  It can
        be None if the the toolkit lacks a native "mouse event" object.

        Returns True if the event has been handled within the Enable object
        hierarchy, or False otherwise.
        """
        if self._size is None:
            # PZW: Hack!
            # We need to handle the cases when the window hasn't been painted
            # yet, but it's gotten a mouse event.  In such a case, we just
            # ignore the mouse event. If the window has been painted, then
            # _size will have some sensible value.
            return False

        mouse_event = self._create_mouse_event(event)
        # if no mouse event generated for some reason, return
        if mouse_event is None:
            return False

        mouse_owner = self.mouse_owner

        if mouse_owner is not None:
            # A mouse_owner has grabbed the mouse.  Check to see if we need to
            # compose a net transform by querying each of the objects in the
            # dispatch history in turn, or if we can just apply a saved
            # top-level transform.
            history = self.mouse_owner_dispatch_history
            if history is not None and len(history) > 0:
                # Assemble all the transforms
                transforms = [c.get_event_transform() for c in history]
                total_transform = reduce(dot, transforms[::-1])
                mouse_event.push_transform(total_transform)
            elif self.mouse_owner_transform is not None:
                mouse_event.push_transform(self.mouse_owner_transform)

            mouse_owner.dispatch(mouse_event, event_name)
            self._pointer_owner = mouse_owner
        else:
            # Normal event handling loop
            if self.overlay is not None:
                # TODO: implement this...
                pass
            if (not mouse_event.handled) and (self.component is not None):
                # Test to see if we need to generate a mouse_leave event
                if self._prev_event_handler:
                    if not self._prev_event_handler.is_in(
                            mouse_event.x, mouse_event.y):
                        self._prev_event_handler.dispatch(
                            mouse_event, "pre_mouse_leave"
                        )
                        mouse_event.handled = False
                        self._prev_event_handler.dispatch(
                            mouse_event, "mouse_leave"
                        )
                        self._prev_event_handler = None

                if self.component.is_in(mouse_event.x, mouse_event.y):
                    # Test to see if we need to generate a mouse_enter event
                    if self._prev_event_handler != self.component:
                        self._prev_event_handler = self.component
                        self.component.dispatch(mouse_event, "pre_mouse_enter")
                        mouse_event.handled = False
                        self.component.dispatch(mouse_event, "mouse_enter")

                    # Fire the actual event
                    self.component.dispatch(mouse_event, "pre_" + event_name)
                    mouse_event.handled = False
                    self.component.dispatch(mouse_event, event_name)

        # If this event requires setting the keyboard focus, set the first
        # component under the mouse pointer that accepts focus as the new focus
        # owner (otherwise, nobody owns the focus):
        if set_focus:
            # If the mouse event was a click, then we set the toolkit's
            # focus to ourselves
            if (mouse_event.left_down
                    or mouse_event.middle_down
                    or mouse_event.right_down
                    or mouse_event.mouse_wheel != 0):
                self._set_focus()

            if (self.component is not None) and (self.component.accepts_focus):
                if self.focus_owner is None:
                    self.focus_owner = self.component
                else:
                    pass

        return mouse_event.handled

    # -------------------------------------------------------------------------
    #  Generic drag event handler:
    # -------------------------------------------------------------------------
    def _handle_drag_event(self, event_name, event, set_focus=False):
        """ **event** should be a toolkit-specific opaque object that will
        be passed in to the backend's _create_drag_event() method.  It can
        be None if the the toolkit lacks a native "drag event" object.

        Returns True if the event has been handled within the Enable object
        hierarchy, or False otherwise.
        """
        if self._size is None:
            # PZW: Hack!
            # We need to handle the cases when the window hasn't been painted
            # yet, but it's gotten a mouse event.  In such a case, we just
            # ignore the mouse event. If the window has been painted, then
            # _size will have some sensible value.
            return False

        drag_event = self._create_drag_event(event)
        # if no mouse event generated for some reason, return
        if drag_event is None:
            return False

        if self.component is not None:
            # Test to see if we need to generate a drag_leave event
            if self._prev_event_handler:
                if not self._prev_event_handler.is_in(
                        drag_event.x, drag_event.y):
                    self._prev_event_handler.dispatch(
                        drag_event, "pre_drag_leave"
                    )
                    drag_event.handled = False
                    self._prev_event_handler.dispatch(drag_event, "drag_leave")
                    self._prev_event_handler = None

            if self.component.is_in(drag_event.x, drag_event.y):
                # Test to see if we need to generate a mouse_enter event
                if self._prev_event_handler != self.component:
                    self._prev_event_handler = self.component
                    self.component.dispatch(drag_event, "pre_drag_enter")
                    drag_event.handled = False
                    self.component.dispatch(drag_event, "drag_enter")

                # Fire the actual event
                self.component.dispatch(drag_event, "pre_" + event_name)
                drag_event.handled = False
                self.component.dispatch(drag_event, event_name)

        return drag_event.handled

    def set_tooltip(self, components):
        "Set the window's tooltip (if necessary)"
        raise NotImplementedError

    def redraw(self):
        """ Requests that the window be redrawn. """
        self._redraw()

    def cleanup(self):
        """ Clean up after ourselves.
        """
        if self.component is not None:
            self.component.cleanup(self)
            self.component.parent = None
            self.component.window = None
            self.component = None

        self.control = None
        if self._gc is not None:
            self._gc.window = None
            self._gc = None

    def _needs_redraw(self, bounds):
        "Determine if a specified region intersects the update region"
        return does_disjoint_intersect_coordinates(
            self._update_region, bounds_to_coordinates(bounds)
        )

    def _paint(self, event=None):
        """ This method is called directly by the UI toolkit's callback
        mechanism on the paint event.
        """
        if self.control is None:
            # the window has gone away, but let the window implementation
            # handle the event as needed
            self._window_paint(event)
            return

        # Create a new GC if necessary
        size = self._get_control_size()
        if (self._size != tuple(size)) or (self._gc is None):
            self._size = tuple(size)
            self._gc = self._create_gc(size)

        # Always give the GC a chance to initialize
        self._init_gc()

        # Layout components and draw
        if hasattr(self.component, "do_layout"):
            self.component.do_layout()
        gc = self._gc
        self.component.draw(gc, view_bounds=(0, 0, size[0], size[1]))

        # damaged_regions = draw_result['damaged_regions']
        # FIXME: consolidate damaged regions if necessary
        if not self.use_damaged_region:
            self._update_region = None

        # Perform a paint of the GC to the window (only necessary on backends
        # that render to an off-screen buffer)
        self._window_paint(event)

        self._update_region = []

    def __getstate__(self):
        attribs = ("component", "bgcolor", "overlay", "_scroll_origin")
        state = {}
        for attrib in attribs:
            state[attrib] = getattr(self, attrib)
        return state

    # -------------------------------------------------------------------------
    # Wire up the mouse event handlers
    # -------------------------------------------------------------------------

    def _on_left_down(self, event):
        self._handle_mouse_event("left_down", event, set_focus=True)

    def _on_left_up(self, event):
        self._handle_mouse_event("left_up", event)

    def _on_left_dclick(self, event):
        self._handle_mouse_event("left_dclick", event)

    def _on_right_down(self, event):
        self._handle_mouse_event("right_down", event, set_focus=True)

    def _on_right_up(self, event):
        self._handle_mouse_event("right_up", event)

    def _on_right_dclick(self, event):
        self._handle_mouse_event("right_dclick", event)

    def _on_middle_down(self, event):
        self._handle_mouse_event("middle_down", event)

    def _on_middle_up(self, event):
        self._handle_mouse_event("middle_up", event)

    def _on_middle_dclick(self, event):
        self._handle_mouse_event("middle_dclick", event)

    def _on_mouse_move(self, event):
        self._handle_mouse_event("mouse_move", event, 1)

    def _on_mouse_wheel(self, event):
        self._handle_mouse_event("mouse_wheel", event)

    def _on_mouse_enter(self, event):
        self._handle_mouse_event("mouse_enter", event)

    def _on_mouse_leave(self, event):
        self._handle_mouse_event("mouse_leave", event, -1)

    # Additional event handlers that are not part of normal Interactors
    def _on_window_enter(self, event):
        # TODO: implement this to generate a mouse_leave on self.component
        pass

    def _on_window_leave(self, event):
        if self._size is None:
            # PZW: Hack!
            # We need to handle the cases when the window hasn't been painted
            # yet, but it's gotten a mouse event.  In such a case, we just
            # ignore the mouse event. If the window has been painted, then
            # _size will have some sensible value.
            self._prev_event_handler = None
        if self._prev_event_handler:
            mouse_event = self._create_mouse_event(event)
            self._prev_event_handler.dispatch(mouse_event, "mouse_leave")
            self._prev_event_handler = None

    # -------------------------------------------------------------------------
    # Wire up the keyboard event handlers
    # -------------------------------------------------------------------------

    def _on_key_pressed(self, event):
        self._handle_key_event("key_pressed", event)

    def _on_key_released(self, event):
        self._handle_key_event("key_released", event)

    def _on_character(self, event):
        self._handle_key_event("character", event)
