""" Defines a Viewport which renders sub-areas of components """

from __future__ import with_statement

# Standard library imports
from numpy import array, dot

# Enthought library traits
from enable.tools.viewport_zoom_tool import ViewportZoomTool
from enable.simple_layout import simple_container_get_preferred_size, \
                                            simple_container_do_layout
from traits.api import (Bool, Delegate, Float, Instance, Enum, List,
        Any, on_trait_change)
from kiva import affine

# Local relative imports
from enable_traits import bounds_trait, coordinate_trait
from base import empty_rectangle, intersect_bounds
from component import Component
from container import Container
from canvas import Canvas


class Viewport(Component):
    """
    A "window" or "view" into a sub-region of another component.
    """

    # The component we are viewing
    component = Instance(Component)

    # The position of our viewport into our component (in the component's
    # coordinate space)
    view_position = coordinate_trait

    # The bounds of our viewport in the space of our component
    view_bounds = bounds_trait

    # Whether or not this viewport should stay constrained to the bounds
    # of the viewed component
    # TODO: Implement this
    stay_inside = Bool(False)

    # Enable Zoom interaction
    enable_zoom = Bool(False)

    # The zoom tool
    zoom_tool = Instance(ViewportZoomTool)

    # Zoom scaling factor for this viewport - Ratio of old bounds to new bounds.
    # Zoom less than 1.0 means we are zoomed out, and more than 1.0 means
    # we are zoomed in.  Zoom should always be positive and nonzero.
    zoom = Float(1.0)

    # Whether to initiate layout on the viewed component.  This is necessary
    # if the component is only viewed through viewports, in which case one
    # of the viewports must lay it out or bounds must be set explicitly
    # on the component.
    initiate_layout = Bool(False)


    min_zoom = Delegate('zoom_tool', modify=True)
    max_zoom = Delegate('zoom_tool', modify=True)

    _component_preferred_size = Any(None)

    #------------------------------------------------------------------------
    # Public methods
    #------------------------------------------------------------------------

    def __init__(self, **traits):
        Component.__init__(self, **traits)
        self._update_component_view_bounds()
        if 'zoom_tool' not in traits:
            self.zoom_tool = ViewportZoomTool(self)
        if self.enable_zoom:
            self._enable_zoom_changed(False, True)
        return

    def components_at(self, x, y, add_containers = False):
        """
        Returns the list of components inside the viewport at the given (x,y)
        in the viewport's native coordinate space (not in the space of the
        component it is viewing).

        Although Viewports are not containers, they support this method.
        """
        if self.is_in(x, y):
            if self.component is not None:
                # Transform (scale + translate) the incoming X and Y
                # coordinates from our coordinate system into the coordinate
                # system of the component we are viewing.
                x_trans, y_trans = self.viewport_to_component(x, y)

                if isinstance(self.component, Container):
                    return self.component.components_at(x_trans, y_trans)
                elif self.component.is_in(x_trans, y_trans):
                    return [self.component]
                else:
                    return []
        else:
            return []

    def invalidate_draw(self, damaged_regions=None, self_relative=False,
                        view_relative=False):
        if view_relative and damaged_regions:
            damaged_regions = [[region[0] - self.view_position[0],
                                region[1] - self.view_position[1],
                                region[2], region[3]] for region in damaged_regions]
        super(Viewport, self).invalidate_draw(damaged_regions=damaged_regions,
                                              self_relative=self_relative)
        return

    def cleanup(self, window):
        """When a window viewing or containing a component is destroyed,
        cleanup is called on the component to give it the opportunity to
        delete any transient state it may have (such as backbuffers)."""
        if self.component:
            self.component.cleanup(window)

    def get_preferred_size(self):
        """If we're initiating layout, act like an OverlayPlotContainer,
           otherwise do the normal component action"""
        if self.initiate_layout:
            self._component_preferred_size = simple_container_get_preferred_size(self, components=[container])
            return self._component_preferred_size
        else:
            return super(Viewport, self).get_preferred_size()

    def viewport_to_component(self, x, y):
        """ Given a coordinate X and Y in the viewport's coordinate system,
        returns and X and Y in the coordinate system of the viewed
        component.
        """
        transform = self.get_event_transform()
        return dot(array([x,y,1]), transform)[:2]

    def component_to_viewport(self, x, y):
        """ Given a coordinate X and Y in the viewed component's coordinate
        system, returns a coordinate in the coordinate system of the
        Viewport.
        """
        vx, vy = self.view_position
        ox, oy = self.outer_position
        newx = (x - vx) * self.zoom + ox
        newy = (y - vx) * self.zoom + oy
        return (newx, newy)


    def get_event_transform(self, event=None, suffix=""):
        transform = affine.affine_identity()

        if isinstance(self.component, Component):
            # If we have zoom enabled, scale events.  Since affine transforms
            # multiply from the left, we build up the transform from the
            # inside of the viewport outwards.
            if self.enable_zoom and self.zoom != 1.0:
                transform = affine.translate(transform, *self.view_position)
                transform = affine.scale(transform, 1/self.zoom, 1/self.zoom)
                transform = affine.translate(transform, -self.outer_position[0],
                                                        -self.outer_position[1])
            else:
                x_offset = self.view_position[0] - self.outer_position[0]
                y_offset = self.view_position[1] - self.outer_position[1]
                transform = affine.translate(transform, x_offset, y_offset)

        return transform



    #------------------------------------------------------------------------
    # Component interface
    #------------------------------------------------------------------------

    def _draw_mainlayer(self, gc, view_bounds=None, mode="normal"):

        # For now, ViewPort ignores the view_bounds that are passed in...
        # Long term, it should be intersected with the view_position to
        # compute a new view_bounds to pass in to our component.
        if self.component is not None:

            x, y = self.position
            view_x, view_y = self.view_position
            with gc:
                # Clip in the viewport's space (screen space).  This ensures
                # that the half-pixel offsets we us are actually screen pixels,
                # and it's easier/more accurate than transforming the clip
                # rectangle down into the component's space (especially if zoom
                # is involved).
                gc.clip_to_rect(x-0.5, y-0.5,
                                self.width+1,
                                self.height+1)
    
                # There is a two-step transformation from the viewport's "outer"
                # coordinates into the coordinates space of the viewed component:
                # scaling, followed by a translation.
                if self.enable_zoom:
                    if self.zoom != 0:
                        gc.scale_ctm(self.zoom, self.zoom)
                        gc.translate_ctm(x/self.zoom - view_x, y/self.zoom - view_y)
                    else:
                        raise RuntimeError("Viewport zoomed out too far.")
                else:
                    gc.translate_ctm(x - view_x, y - view_y)
    
                # Now transform the passed-in view_bounds; this is not the same thing as
                # self.view_bounds!
                if view_bounds:
                    # Find the intersection rectangle of the viewport with the view_bounds,
                    # and transform this into the component's space.
                    clipped_view = intersect_bounds(self.position + self.bounds, view_bounds)
                    if clipped_view != empty_rectangle:
                        # clipped_view and self.position are in the space of our parent
                        # container.  we know that self.position -> view_x,view_y
                        # in the coordinate space of our component.  So, find the
                        # vector from self.position to clipped_view, then add this to
                        # view_x and view_y to generate the transformed coordinates
                        # of clipped_view in our component's space.
                        offset = array(clipped_view[:2]) - array(self.position)
                        new_bounds = ((offset[0]/self.zoom + view_x),
                                      (offset[1]/self.zoom + view_y),
                                      clipped_view[2] / self.zoom, clipped_view[3] / self.zoom)
                        self.component.draw(gc, new_bounds, mode=mode)

        return

    def _do_layout(self):
        if self.initiate_layout:
            self.component.bounds = list(self.component.get_preferred_size())
            self.component.do_layout()

        else:
            super(Viewport, self)._do_layout()
        return

    def _dispatch_stateful_event(self, event, suffix):
        if isinstance(self.component, Component):
            transform = self.get_event_transform(event, suffix)
            event.push_transform(transform, caller=self)
            try:
                self.component.dispatch(event, suffix)
            finally:
                event.pop(caller=self)

        return


    #------------------------------------------------------------------------
    # Event handlers
    #------------------------------------------------------------------------


    def _enable_zoom_changed(self, old, new):
        """
            Add or remove the zoom tool overlay depending whether
            or not zoom is enabled.
        """
        if self.zoom_tool is None:
            return

        if self.enable_zoom:
            if not self.zoom_tool in self.tools:
                self.tools.append(self.zoom_tool)
        else:
            if self.zoom_tool in self.tools:
                self.tools.remove(self.zoom_tool)

    def _update_component_view_bounds(self):
        """ Updates the optional .view_bounds trait on our component;
        mostly used for Canvas objects.
        """
        if isinstance(self.component, Canvas):
            llx, lly = self.view_position
            self.component.view_bounds = (llx, lly,
                                          llx + self.view_bounds[0]-1,
                                          lly + self.view_bounds[1]-1)
        return

    def _component_changed(self, old, new):
        if (old is not None) and (self in old.viewports):
            old.viewports.remove(self)

        if (new is not None) and (self not in new.viewports):
            new.viewports.append(self)
            self._update_component_view_bounds()
        return

    def _bounds_changed(self, old, new):
        Component._bounds_changed(self, old, new)
        self.set(view_bounds = [new[0]/self.zoom, new[1]/self.zoom],
                 trait_change_notify=False)
        self._update_component_view_bounds()
        return

    def _bounds_items_changed(self, event):
        return self._bounds_changed(None, self.bounds)

    @on_trait_change("view_bounds,view_position")
    def _handle_view_box_changed(self):
        self._update_component_view_bounds()

    def _get_position(self):
        return self.view_position

    def _get_bounds(self):
        return self.view_bounds

# EOF

