# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
""" Defines the basic Container class """

# Major library imports
import warnings

# Enthought library imports
from kiva import affine
from traits.api import Bool, Enum, Instance, List, Property, Tuple

# Local, relative imports
from .base import empty_rectangle, intersect_bounds
from .component import Component
from .events import BlobEvent, BlobFrameEvent, DragEvent, MouseEvent


class Container(Component):
    """ A Container is a logical container that holds other Components within
    it and provides an origin for Components to position themselves. Containers
    can be "nested" (although "overlayed" is probably a better term).

    If auto_size is True, the container will automatically update its bounds to
    enclose all of the components handed to it, so that a container's bounds
    serve as abounding box (although not necessarily a minimal bounding box) of
    its contained components.
    """

    # The list of components within this frame
    components = Property  # List(Component)

    # Whether or not the container should automatically maximize itself to
    # fit inside the Window, if this is a top-level container.
    #
    # NOTE: the way that a Container determines that it's a top-level window is
    # that someone has explicitly set its .window attribute. If you need to do
    # this for some other reason, you may want to turn fit_window off.
    fit_window = Bool(True)

    # If true, the container get events before its children.  Otherwise, it
    # gets them afterwards.
    intercept_events = Bool(True)

    # Dimensions in which this container can resize to fit its components.
    # This trait only applies to dimensions that are also resizable; if the
    # container is not resizable in a certain dimension, then fit_components
    # has no effect.
    #
    # Also, note that the container does not *automatically* resize itself
    # based on the value of this trait.  Rather, this trait determines
    # what value is reported in get_preferred_size(); it is up to the parent
    # of this container to make sure that it is allocated the size that it
    # needs by setting its bounds appropriately.
    #
    # TODO: Merge resizable and this into a single trait?  Or have a separate
    # "fit" flag for each dimension in the **resizable** trait?
    # TODO: This trait is used in layout methods of various Container
    # subclasses in Chaco.  We need to move those containers into
    # Enable.
    fit_components = Enum("", "h", "v", "hv")

    # Whether or not the container should auto-size itself to fit all of its
    # components.
    # Note: This trait is still used, but will be eventually removed in favor
    # of **fit_components**.
    auto_size = Bool(False)

    # The default size of this container if it is empty.
    default_size = Tuple(0, 0)

    # The layers that the container will draw first, so that they appear
    # under the component layers of the same name.
    container_under_layers = Tuple(
        "background", "image", "underlay", "mainlayer"
    )

    # ------------------------------------------------------------------------
    # Private traits
    # ------------------------------------------------------------------------

    # Shadow trait for self.components
    _components = List  # List(Component)

    # Set of components that last handled a mouse event.  We keep track of
    # this so that we can generate mouse_enter and mouse_leave events of
    # our own.
    _prev_event_handlers = Instance(set, ())

    # This container can render itself in a different mode than what it asks of
    # its contained components.  This attribute stores the rendering mode that
    # this container requests of its children when it does a _draw(). If the
    # attribute is set to "default", then whatever mode is handed in to _draw()
    # is used.
    _children_draw_mode = Enum("default", "normal", "overlay", "interactive")

    # ------------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------------

    def __init__(self, *components, **traits):
        Component.__init__(self, **traits)
        for component in components:
            self.add(component)
        if "bounds" in traits and "auto_size" not in traits:
            self.auto_size = False

        if "intercept_events" in traits:
            warnings.warn(
                "'intercept_events' is a deprecated trait",
                warnings.DeprecationWarning,
            )

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

    def remove(self, *components):
        """ Removes components from this container """
        for component in components:
            if component in self._components:
                component.container = None
                self._components.remove(component)
            else:
                raise RuntimeError(
                    "Unable to remove component from container."
                )

            # Check to see if we need to compact.
            if self.auto_size:
                if ((component.outer_x2 == self.width)
                        or (component.outer_y2 == self.height)
                        or (component.x == 0)
                        or (component.y == 0)):
                    self.compact()

        self.invalidate_draw()

    def insert(self, index, component):
        "Inserts a component into a specific position in the components list"
        if component.container is not None:
            component.container.remove(component)
        component.container = self
        self._components.insert(index, component)

        self.invalidate_draw()

    def components_at(self, x, y):
        """
        Returns a list of the components underneath the given point (given in
        the parent coordinate frame of this container).
        """
        result = []
        if self.is_in(x, y):
            xprime = x - self.position[0]
            yprime = y - self.position[1]
            for component in self._components[::-1]:
                if component.is_in(xprime, yprime):
                    result.append(component)
        return result

    def raise_component(self, component):
        """ Raises the indicated component to the top of the Z-order """
        c = self._components
        ndx = c.index(component)
        if len(c) > 1 and ndx != len(c) - 1:
            self._components = c[:ndx] + c[ndx + 1:] + [component]

    def lower_component(self, component):
        """ Puts the indicated component to the very bottom of the Z-order """
        raise NotImplementedError

    def cleanup(self, window):
        """When a window viewing or containing a component is destroyed,
        cleanup is called on the component to give it the opportunity to
        delete any transient state it may have (such as backbuffers)."""
        if self._components:
            for component in self._components:
                component.cleanup(window)

    def compact(self):
        """
        Causes this container to update its bounds to be a compact bounding
        box of its components.  This may cause the container to recalculate
        and adjust its position relative to its parent container (and adjust
        the positions of all of its contained components accordingly).
        """
        # Loop over our components and determine the bounding box of all of
        # the components.
        ll_x, ll_y, ur_x, ur_y = self._calc_bounding_box()
        if len(self._components) > 0:
            # Update our position and the positions of all of our components,
            # but do it quietly
            for component in self._components:
                component.trait_setq(
                    position=[component.x - ll_x, component.y - ll_y]
                )

            # Change our position (in our parent's coordinate frame) and
            # update our bounds
            self.position = [self.x + ll_x, self.y + ll_y]

        self.bounds = [ur_x - ll_x, ur_y - ll_y]

    # ------------------------------------------------------------------------
    # Protected methods
    # ------------------------------------------------------------------------

    def _calc_bounding_box(self):
        """
        Returns a 4-tuple (x,y,x2,y2) of the bounding box of all our contained
        components.  Expressed as coordinates in our local coordinate frame.
        """
        if len(self._components) == 0:
            return (0.0, 0.0, 0.0, 0.0)
        else:
            comp = self._components[0]
            ll_x = comp.outer_x
            ll_y = comp.outer_y
            ur_x = comp.outer_x2
            ur_y = comp.outer_y2

        for component in self._components[1:]:
            if component.x < ll_x:
                ll_x = component.x
            if component.x2 > ur_x:
                ur_x = component.x2
            if component.y < ll_y:
                ll_y = component.y
            if component.y2 > ur_y:
                ur_y = component.y2
        return (ll_x, ll_y, ur_x, ur_y)

    def _dispatch_draw(self, layer, gc, view_bounds, mode):
        """ Renders the named *layer* of this component.
        """
        new_bounds = self._transform_view_bounds(view_bounds)
        if new_bounds == empty_rectangle:
            return

        if self.layout_needed:
            self.do_layout()

        # Give the container a chance to draw first for the layers that are
        # considered "under" or "at" the main layer level
        if layer in self.container_under_layers:
            my_handler = getattr(self, "_draw_container_" + layer, None)
            if my_handler:
                my_handler(gc, view_bounds, mode)

        # Now transform coordinates and draw the children
        visible_components = self._get_visible_components(new_bounds)
        if visible_components:
            with gc:
                gc.translate_ctm(*self.position)
                for component in visible_components:
                    if component.unified_draw:
                        # Plot containers that want unified_draw only get
                        # called if their draw_layer matches the current layer
                        # we're rendering
                        if component.draw_layer == layer:
                            component._draw(gc, new_bounds, mode)
                    else:
                        component._dispatch_draw(layer, gc, new_bounds, mode)

        # The container's annotation and overlay layers draw over those of
        # its components.
        # FIXME: This needs to be abstracted so that when subclasses override
        # the draw_order list, these are pulled from the subclass list instead
        # of hardcoded here.
        if layer in ("annotation", "overlay", "border"):
            my_handler = getattr(self, "_draw_container_" + layer, None)
            if my_handler:
                my_handler(gc, view_bounds, mode)

    def _draw_container(self, gc, mode="default"):
        "Draw the container background in a specified graphics context"
        pass

    def _draw_container_background(self, gc, view_bounds=None, mode="normal"):
        self._draw_background(gc, view_bounds, mode)

    def _draw_container_overlay(self, gc, view_bounds=None, mode="normal"):
        self._draw_overlay(gc, view_bounds, mode)

    def _draw_container_underlay(self, gc, view_bounds=None, mode="normal"):
        self._draw_underlay(gc, view_bounds, mode)

    def _draw_container_border(self, gc, view_bounds=None, mode="normal"):
        self._draw_border(gc, view_bounds, mode)

    def _get_visible_components(self, bounds):
        """ Returns a list of this plot's children that are in the bounds. """
        if bounds is None:
            return [c for c in self.components if c.visible]

        visible_components = []
        for component in self.components:
            if not component.visible:
                continue
            tmp = intersect_bounds(
                component.outer_position + component.outer_bounds, bounds
            )
            if tmp != empty_rectangle:
                visible_components.append(component)
        return visible_components

    def _should_layout(self, component):
        """ Returns True if it is appropriate for the container to lay out
        the component; False if not.
        """
        if (not component
                or (not component.visible and not component.invisible_layout)):
            return False
        else:
            return True

    def _should_compact(self):
        """ Returns True if the container needs to call compact().  Subclasses
        can overload this method as needed.
        """
        if self.auto_size:
            width = self.width
            height = self.height
            for component in self.components:
                x, y = component.outer_position
                x2 = component.outer_x2
                y2 = component.outer_y2
                if (x2 >= width) or (y2 >= height) or (x < 0) or (y < 0):
                    return True
        else:
            return False

    def _transform_view_bounds(self, view_bounds):
        """
        Transforms the given view bounds into our local space and computes a
        new region that can be handed off to our children.  Returns a 4-tuple
        of the new position+bounds, or None (if None was passed in), or the
        value of empty_rectangle (from enable.base) if the intersection
        resulted in a null region.
        """
        if view_bounds:
            # Check if we are visible
            tmp = intersect_bounds(self.position + self.bounds, view_bounds)
            if tmp == empty_rectangle:
                return empty_rectangle
            # Compute new_bounds, which is the view_bounds transformed into
            # our coordinate space
            v = view_bounds
            new_bounds = (v[0] - self.x, v[1] - self.y, v[2], v[3])
        else:
            new_bounds = None
        return new_bounds

    def _component_bounds_changed(self, component):
        "Called by contained objects when their bounds change"
        # For now, just punt and call compact()
        if self.auto_size:
            self.compact()

    def _component_position_changed(self, component):
        "Called by contained objects when their position changes"
        # For now, just punt and call compact()
        if self.auto_size:
            self.compact()

    # ------------------------------------------------------------------------
    # Deprecated interface
    # ------------------------------------------------------------------------

    def _draw_overlays(self, gc, view_bounds=None, mode="normal"):
        """ Method for backward compatability with old drawing scheme.
        """
        warnings.warn("Containter._draw_overlays is deprecated.")
        for component in self.overlays:
            component.overlay(component, gc, view_bounds, mode)

    # ------------------------------------------------------------------------
    # Property setters & getters
    # ------------------------------------------------------------------------

    def _get_components(self):
        return self._components

    def _set_components(self, new):
        self._components = new

    def _get_layout_needed(self):
        # Override the parent implementation to take into account whether any
        # of our contained components need layout.
        if self._layout_needed:
            return True
        else:
            for c in self.components:
                if c.layout_needed:
                    return True
            else:
                return False

    # ------------------------------------------------------------------------
    # Interactor interface
    # ------------------------------------------------------------------------

    def normal_mouse_leave(self, event):
        event.push_transform(self.get_event_transform(event), caller=self)
        for component in self._prev_event_handlers:
            component.dispatch(event, "mouse_leave")
        self._prev_event_handlers = set()
        event.pop(caller=self)

    def _container_handle_mouse_event(self, event, suffix):
        """
        This method allows the container to handle a mouse event before its
        children get to see it.  Once the event gets handled, its .handled
        should be set to True, and contained components will not be called
        with the event.
        """
        # super(Container, self)._dispatch_stateful_event(event, suffix)
        Component._dispatch_stateful_event(self, event, suffix)

    def get_event_transform(self, event=None, suffix=""):
        return affine.affine_from_translation(-self.x, -self.y)

    def _dispatch_stateful_event(self, event, suffix):
        """
        Dispatches a mouse event based on the current event_state.  Overrides
        the default Interactor._dispatch_stateful_event by adding some default
        behavior to send all events to our contained children.

        "suffix" is the name of the mouse event as a suffix to the event state
        name, e.g. "_left_down" or "_window_enter".
        """
        if not event.handled:
            if isinstance(event, BlobFrameEvent):
                # This kind of event does not have a meaningful location. Just
                # let all of the child components see it.
                for component in self._components[::-1]:
                    component.dispatch(event, suffix)
                return

            components = self.components_at(event.x, event.y)

            # Translate the event's location to be relative to this container
            event.push_transform(
                self.get_event_transform(event, suffix), caller=self
            )

            try:
                new_component_set = set(components)

                # For "real" mouse events (i.e., not pre_mouse_* events),
                # notify the previous listening components of a mouse or
                # drag leave
                if not suffix.startswith("pre_"):
                    components_left = (
                        self._prev_event_handlers - new_component_set
                    )
                    if components_left:
                        leave_event = None
                        if isinstance(event, MouseEvent):
                            leave_event = event
                            leave_suffix = "mouse_leave"
                        elif isinstance(event, DragEvent):
                            leave_event = event
                            leave_suffix = "drag_leave"
                        elif isinstance(event, (BlobEvent, BlobFrameEvent)):
                            # Do not generate a 'leave' event.
                            pass
                        else:
                            # TODO: think of a better way to handle this rare
                            # case?
                            leave_event = MouseEvent(
                                x=event.x, y=event.y, window=event.window
                            )
                            leave_suffix = "mouse_leave"

                        if leave_event is not None:
                            for component in components_left:
                                component.dispatch(
                                    leave_event, "pre_" + leave_suffix
                                )
                                component.dispatch(leave_event, leave_suffix)
                                event.handled = False

                    # Notify new components of a mouse enter, if the event is
                    # not a mouse_leave or a drag_leave
                    if suffix not in ("mouse_leave", "drag_leave"):
                        components_entered = (
                            new_component_set - self._prev_event_handlers
                        )
                        if components_entered:
                            enter_event = None
                            if isinstance(event, MouseEvent):
                                enter_event = event
                                enter_suffix = "mouse_enter"
                            elif isinstance(event, DragEvent):
                                enter_event = event
                                enter_suffix = "drag_enter"
                            elif isinstance(event, (BlobEvent, BlobFrameEvent)):  # noqa: E501
                                # Do not generate an 'enter' event.
                                pass
                            if enter_event:
                                for component in components_entered:
                                    component.dispatch(
                                        enter_event, "pre_" + enter_suffix
                                    )
                                    component.dispatch(
                                        enter_event, enter_suffix
                                    )
                                    event.handled = False

                # Handle the actual event
                # Only add event handlers to the list of previous event
                # handlers if they actually receive the event (and the event
                # is not a pre_* event).
                if not suffix.startswith("pre_"):
                    self._prev_event_handlers = set()
                for component in components:
                    component.dispatch(event, suffix)
                    if not suffix.startswith("pre_"):
                        self._prev_event_handlers.add(component)
                    if event.handled:
                        break
            finally:
                event.pop(caller=self)

            if not event.handled:
                self._container_handle_mouse_event(event, suffix)

    # ------------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------------

    def _auto_size_changed(self, old, new):
        # For safety, re-compute our bounds
        if new is True:
            self.compact()
        else:
            pass

    def _window_resized(self, newsize):
        if newsize is not None:
            self.bounds = [newsize[0] - self.x, newsize[1] - self.y]

    # FIXME: Need a _window_changed to remove this handler if the window
    # changes

    def _fit_window_changed(self, old, new):
        if self._window is not None:
            if not self.fit_window:
                self._window.on_trait_change(
                    self._window_resized, "resized", remove=True
                )
            else:
                self._window.on_trait_change(self._window_resized, "resized")

    def _bounds_changed(self, old, new):
        # crappy... calling our parent's handler seems like a common traits
        # event handling problem
        super(Container, self)._bounds_changed(old, new)
        self._layout_needed = True
        self.invalidate_draw()

    def _bounds_items_changed(self, event):
        super(Container, self)._bounds_items_changed(event)
        self._layout_needed = True
        self.invalidate_draw()

    def _bgcolor_changed(self):
        self.invalidate_draw()
        self.request_redraw()

    def __components_items_changed(self, event):
        self._layout_needed = True

    def __components_changed(self, event):
        self._layout_needed = True
        self.invalidate_draw()

    # -------------------------------------------------------------------------
    # Old / deprecated draw methods; here for backwards compatibility
    # -------------------------------------------------------------------------

    def _draw_component(self, gc, view_bounds=None, mode="normal"):
        """ Draws the component.

        This method is preserved for backwards compatibility. Overrides
        the implementation in Component.
        """
        with gc:
            gc.set_antialias(False)

            self._draw_container(gc, mode)
            self._draw_background(gc, view_bounds, mode)
            self._draw_underlay(gc, view_bounds, mode)
            self._draw_children(
                gc, view_bounds, mode
            )  # This was children_draw_mode
            self._draw_overlays(gc, view_bounds, mode)

    def _draw_children(self, gc, view_bounds=None, mode="normal"):

        new_bounds = self._transform_view_bounds(view_bounds)
        if new_bounds == empty_rectangle:
            return

        with gc:
            gc.set_antialias(False)
            gc.translate_ctm(*self.position)
            for component in self.components:
                if new_bounds:
                    tmp = intersect_bounds(
                        component.outer_position + component.outer_bounds,
                        new_bounds,
                    )
                    if tmp == empty_rectangle:
                        continue
                with gc:
                    component.draw(gc, new_bounds, mode)
