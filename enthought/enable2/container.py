""" Defines the basic Container class """

# Major library imports
import warnings

# Enthought library imports
from enthought.traits.api import Any, Bool, Enum, HasTraits, Instance, List, \
        Property, Trait, Tuple

# Local, relative imports
from base import empty_rectangle, intersect_bounds
from component import Component
from enable_traits import border_size_trait
from events import BlobEvent, DragEvent, MouseEvent
from abstract_layout_controller import AbstractLayoutController


class AbstractResolver(HasTraits):
    """
    A Resolver traverses a component DB and matches a specifier.
    """

    def match(self, db, query):
        """ Queries a component DB using a dict of keyword-val conditions.
        Each resolver defines its set of allowed keywords.
        """
        raise NotImplementedError


class DefaultResolver(AbstractResolver):
    """
    Basic resolver that searches a container's DB of components using the
    following conditions:

        id=foo :  the component's .id must be 'foo'

        class=['foo','bar'] :  the component's .class must be in the list

        target='foo' :  the component's .target is 'foo'; this usually applies
                        to tools, overlays, and decorators
    """

    def match(self, db, query):
        pass


class Container(Component):
    """
    A Container is a logical container that holds other Components within it and
    provides an origin for Components to position themselves.  Containers can
    be "nested" (although "overlayed" is probably a better term).

    If auto_size is True, the container will automatically update its bounds to
    enclose all of the components handed to it, so that a container's bounds
    serve as abounding box (although not necessarily a minimal bounding box) of
    its contained components.
    """

    # The list of components within this frame
    components = Property    # List(Component)

    # Whether or not the container should auto-size itself to fit all of its
    # components.
    auto_size = Bool(False)

    # Whether or not the container should automatically maximize itself to
    # fit inside the Window, if this is a top-level container.
    # Note: the way that a Container determines that it's a top-level
    #       window is that someone has explicitly set its .window attribute.
    #       If you need to do this for some other reason, you may want to
    #       turn fit_window off.
    fit_window = Bool(True)

    # If true, the container get events before its children.  Otherwise, it
    # gets them afterwards.
    intercept_events = Bool(True)

    # Dimensions in which this container can resize to fit its components.
    # This is similar to the **resizable** trait on PlotComponent. Chaco
    # plot containers use this attribute in preference to the Enable
    # **auto_size** attribute (which is overridden to be False by default).
    fit_components = Enum("", "h", "v", "hv") # TODO: remove the auto_size
                                              # trait from enable's Container

    # The default size of this container if it is empty.
    default_size = Tuple(0, 0)

    # The layers that the container will draw first, so that they appear
    # under the component layers of the same name.
    container_under_layers = Tuple("background", "image", "underlay", "mainlayer")

    #------------------------------------------------------------------------
    # DOM-related traits
    # (Note: These are unused as of 8/13/2007)
    #------------------------------------------------------------------------

    # The layout controller determines how the container's internal layout
    # mechanism works.  It can perform the actual layout or defer to an
    # enclosing container's layout controller.  The default controller is
    # a cooperative/recursive layout controller.
    layout_controller = Instance(AbstractLayoutController)

    # This object resolves queries for components
    resolver = Instance(AbstractResolver)


    #------------------------------------------------------------------------
    # Private traits
    #------------------------------------------------------------------------

    # Shadow trait for self.components
    _components = List    # List(Component)

    # Set of components that last handled a mouse event.  We keep track of
    # this so that we can generate mouse_enter and mouse_leave events of
    # our own.
    _prev_event_handlers = Instance( set, () )

    # Used by the resolver to cache previous lookups
    _lookup_cache = Any
    
    # This container can render itself in a different mode than what it asks of
    # its contained components.  This attribute stores the rendering mode that
    # this container requests of its children when it does a _draw(). If the
    # attribute is set to "default", then whatever mode is handed in to _draw()
    # is used.
    _children_draw_mode = Enum("default", "normal", "overlay", "interactive")

    #------------------------------------------------------------------------
    # Public methods
    #------------------------------------------------------------------------

    def __init__(self, *components, **traits):
        Component.__init__(self, **traits)
        for component in components:
            self.add(component)
        if "bounds" in traits.keys() and "auto_size" not in traits.keys():
            self.auto_size = False

        if 'intercept_events' in traits:
            warnings.warn("'intercept_events' is a deprecated trait",
                    warnings.DeprecationWarning)
        return

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

    def remove(self, *components):
        """ Removes components from this container """
        for component in components:
            if component in self._components:
                component.container = None
                self._components.remove(component)
            else:
                raise RuntimeError, "Unable to remove component from container."

        self.invalidate_draw()
        return


    def insert(self, index, component):
        "Inserts a component into a specific position in the components list"
        if component.container is not None:
            component.container.remove(component)
        component.container = self
        self._components.insert(index, component)

        self.invalidate_draw()

        return

    def components_at(self, x, y):
        """
        Returns a list of the components underneath the given point (given in
        the parent coordinate frame of this container).
        """
        result = []
        if self.is_in(x,y):
            xprime = x - self.position[0]
            yprime = y - self.position[1]
            for component in self._components[::-1]:
                if component.is_in(xprime, yprime):
                    result.append(component)
        return result

    def get(self, **kw):
        """
        Allows for querying of this container's components.
        """
        # TODO: cache requests
        return self.resolver.query(self._components, kw)

    def cleanup(self, window):
        """When a window viewing or containing a component is destroyed,
        cleanup is called on the component to give it the opportunity to
        delete any transient state it may have (such as backbuffers)."""
        if self._components:
            for component in self._components:
                component.cleanup(window)
        return

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
                component.set(position = [component.x-ll_x, component.y-ll_y],
                              trait_change_notify = False)

            # Change our position (in our parent's coordinate frame) and
            # update our bounds
            self.position = [self.x + ll_x, self.y + ll_y]

        self.bounds = [ur_x - ll_x, ur_y - ll_y]
        return

    #------------------------------------------------------------------------
    # Protected methods
    #------------------------------------------------------------------------

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

    def _draw_children(self, gc, view_bounds=None, mode="normal"):
        
        new_bounds = self._transform_view_bounds(view_bounds)
        if new_bounds == empty_rectangle:
            return
        
        gc.save_state()
        try:
            gc.set_antialias(False)
            gc.translate_ctm(*self.position)
            for component in self.components:
                if new_bounds:
                    tmp = intersect_bounds(component.outer_position + 
                                           component.outer_bounds, new_bounds)
                    if tmp == empty_rectangle:
                        continue
                
                gc.save_state()
                try:
                    component.draw(gc, new_bounds, mode)
                finally:
                    gc.restore_state()
        finally:
            gc.restore_state()
        return

    def _draw_component(self, gc, view_bounds=None, mode="normal"):
        """ Draws the component.

        This method is preserved for backwards compatibility. Overrides 
        the implementation in Component.
        """
        gc.save_state()
        gc.set_antialias(False)

        self._draw_container(gc, mode)
        self._draw_background(gc, view_bounds, mode)
        self._draw_underlay(gc, view_bounds, mode)
        self._draw_children(gc, view_bounds, mode) #This was children_draw_mode
        self._draw_overlays(gc, view_bounds, mode)
        gc.restore_state()
        return
    
        # The container's annotation and overlay layers draw over those of 
        # its components.
        if layer in ("annotation", "overlay"):
            my_handler = getattr(self, "_draw_container_" + layer, None)
            if my_handler:
                my_handler(gc, view_bounds, mode)
        
        return

    def _draw_container(self, gc, mode="default"):
        "Draw the container background in a specified graphics context"
        pass
        
    def _draw_container_background(self, gc, view_bounds=None, mode="normal"):
        Component._draw_background(self, gc, view_bounds, mode)
        return
        
    def _draw_container_overlay(self, gc, view_bounds=None, mode="normal"):
        Component._draw_overlay(self, gc, view_bounds, mode)
        return

    def _draw_container_underlay(self, gc, view_bounds=None, mode="normal"):
        Component._draw_underlay(self, gc, view_bounds, mode)
        return

    def _get_visible_components(self, bounds):
        """ Returns a list of this plot's children that are in the bounds. """
        if bounds is None:
            return self.components
        
        visible_components = []
        for component in self.components:
            if not component.visible:
                continue
            tmp = intersect_bounds(component.outer_position + 
                                   component.outer_bounds, bounds)
            if tmp != empty_rectangle:
                visible_components.append(component)
        return visible_components

    def _should_layout(self, component):
        """ Returns True if it is appropriate for the container to lay out
        the component; False if not.
        """
        if not component or \
            (not component.visible and not component.invisible_layout):
            return False
        else:
            return True

    def _should_compact(self):
        """ Returns True if the container needs to call compact().  Subclasses
        can overload this method as needed.
        """
        if self.auto_size:
            for component in self.components:
                if (component.outer_x2 >= self.width) or \
                   (component.outer_y2 >= self.height) or \
                   (component.outer_x < 0) or (component.outer_y < 0):
                    return True
        else:
            return False

    def _transform_view_bounds(self, view_bounds):
        """
        Transforms the given view bounds into our local space and computes a new
        region that can be handed off to our children.  Returns a 4-tuple of
        the new position+bounds, or None (if None was passed in), or the value
        of empty_rectangle (from enable2.base) if the intersection resulted
        in a null region.
        """
        if view_bounds:
            # Check if we are visible
            tmp = intersect_bounds(self.position + self.bounds, view_bounds)
            if tmp == empty_rectangle:
                return empty_rectangle
            # Compute new_bounds, which is the view_bounds transformed into
            # our coordinate space
            v = view_bounds
            new_bounds = (v[0]-self.x, v[1]-self.y, v[2], v[3])
        else:
            new_bounds = None
        return new_bounds

    def _component_bounds_changed(self, component):
        "Called by contained objects when their bounds change"
        # For now, just punt and call compact()
        if self.auto_size:
            self.compact()
        return 
    
    def _component_position_changed(self, component):
        "Called by contained objects when their position changes"
        # For now, just punt and call compact()
        if self.auto_size:
            self.compact()
        return

    #------------------------------------------------------------------------
    # Deprecated interface
    #------------------------------------------------------------------------

    def _draw_overlays(self, gc, view_bounds=None, mode="normal"):
        """ Method for backward compatability with old drawing scheme.
        """
        warnings.warn("Containter._draw_overlays is deprecated.")
        for component in self.overlays:
            component.overlay(component, gc, view_bounds, mode)
        return

    #------------------------------------------------------------------------
    # Component interface
    #------------------------------------------------------------------------
    def _dispatch_draw(self, layer, gc, view_bounds, mode):
        """ Renders the named *layer* of this component. 
        """
        new_bounds = self._transform_view_bounds(view_bounds)
        if new_bounds == empty_rectangle:
            return
        
        if self._layout_needed:
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
            gc.save_state()
            try:
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
            finally:
                gc.restore_state()
        
        # The container's annotation and overlay layers draw over those of 
        # its components.
        if layer in ("annotation", "overlay"):
            my_handler = getattr(self, "_draw_container_" + layer, None)
            if my_handler:
                my_handler(gc, view_bounds, mode)
        
        return


    #------------------------------------------------------------------------
    # Property setters & getters
    #------------------------------------------------------------------------

    def _get_components(self):
        return self._components

    #------------------------------------------------------------------------
    # Interactor interface
    #------------------------------------------------------------------------

    def normal_mouse_leave(self, event):
        event.offset_xy(*self.position)
        for component in self._prev_event_handlers:
            component.dispatch(event, "mouse_leave")
        self._prev_event_handlers = set()
        event.pop()
        return

    def _container_handle_mouse_event(self, event, suffix):
        """
        This method allows the container to handle a mouse event before its
        children get to see it.  Once the event gets handled, its .handled
        should be set to True, and contained components will not be called
        with the event.
        """
        super(Container, self)._dispatch_stateful_event(event, suffix)

    def _dispatch_stateful_event(self, event, suffix):
        """
        Dispatches a mouse event based on the current event_state.  Overrides
        the default Interactor._dispatch_stateful_event by adding some default
        behavior to send all events to our contained children.

        "suffix" is the name of the mouse event as a suffix to the event state
        name, e.g. "_left_down" or "_window_enter".
        """

        if not event.handled:
            components = self.components_at(event.x, event.y)

            # Translate the event's location to be relative to this container
            event.offset_xy(*self.position)

            try:
                new_component_set = set(components)

                # For "real" mouse events (i.e., not pre_mouse_* events),
                # notify the previous listening components of a mouse or
                # drag leave
                if not suffix.startswith("pre_"):
                    components_left = self._prev_event_handlers - new_component_set
                    if components_left:
                        if isinstance(event, MouseEvent):
                            leave_event = event
                            leave_suffix = "mouse_leave"
                        elif isinstance(event, DragEvent):
                            leave_event = event
                            leave_suffix = "drag_leave"
                        #elif isinstance(event, BlobEvent):
                        #    leave_event = event
                        #    leave_suffix = "blob_leave"
                        else:
                            # TODO: think of a better way to handle this rare case?
                            leave_event = MouseEvent(x=event.x, y=event.y,
                                                     window=event.window)
                            leave_suffix = "mouse_leave"

                        for component in components_left:
                            component.dispatch(leave_event, "pre_" + leave_suffix)
                            component.dispatch(leave_event, leave_suffix)
                        event.handled = False

                    # Notify new components of a mouse enter, if the event is
                    # not a mouse_leave or a drag_leave
                    if suffix not in ("mouse_leave", "drag_leave"):
                        components_entered = \
                            new_component_set - self._prev_event_handlers
                        if components_entered:
                            enter_event = None
                            if isinstance(event, MouseEvent):
                                enter_event = event
                                enter_suffix = "mouse_enter"
                            elif isinstance(event, DragEvent):
                                enter_event = event
                                enter_suffix = "drag_enter"
                            #elif isinstance(event, BlobEvent):
                            #    enter_event = event
                            #    enter_suffix = "blob_enter"
                            if enter_event:
                                for component in components_entered:
                                    component.dispatch(enter_event, "pre_" + enter_suffix)
                                    component.dispatch(enter_event, enter_suffix)
                                    event.handled = False

                # Handle the actual event
                # Only add event handlers to the list of previous event handlers
                # if they actually receive the event (and the event is not a
                # pre_* event.
                if not suffix.startswith("pre_"):
                    self._prev_event_handlers = set()
                for component in components:
                    component.dispatch(event, suffix)
                    if not suffix.startswith("pre_"):
                        self._prev_event_handlers.add(component)
                    if event.handled:
                        break
            finally:
                event.pop()

            if not event.handled:
                self._container_handle_mouse_event(event, suffix)

        return


    
    #------------------------------------------------------------------------
    # Event handlers
    #------------------------------------------------------------------------

    def _auto_size_changed(self, old, new):
        # For safety, re-compute our bounds
        if new == True:
            self.compact()
        else:
            pass
        return

    def _window_resized(self, newsize):
        if newsize is not None:
            self.bounds = [newsize[0]-self.x, newsize[1]-self.y]
        return


    #FIXME: Need a _window_changed to remove this handler if the window changes

    def _fit_window_changed(self, old, new):
        if self._window is not None:
            if not self.fit_window:
                self._window.on_trait_change(self._window_resized,
                                             "resized", remove=True)
            else:
                self._window.on_trait_change(self._window_resized, "resized")
        return

    def _bounds_changed(self, old, new):
        # crappy... calling our parent's handler seems like a common traits
        # event handling problem
        super(Container, self)._bounds_changed(old, new)
        self._layout_needed = True
        self.invalidate_draw()
        return

    def _bounds_items_changed(self, event):
        super(Container, self)._bounds_items_changed(event)
        self._layout_needed = True
        self.invalidate_draw()
        return

    def _bgcolor_changed(self):
        self.invalidate_draw()
        self.request_redraw()
        return
    def __components_items_changed(self, event):
        self._layout_needed = True
        return

    def __components_changed(self, event):
        self._layout_needed = True
        self.invalidate_draw()
        return

### EOF
