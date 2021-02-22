# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
""" Defines a Viewport which renders sub-areas of components """

# Standard library imports
from numpy import array, dot

# Enthought library traits
from enable.simple_layout import simple_container_get_preferred_size
from enable.tools.viewport_zoom_tool import ViewportZoomTool
from kiva import affine
from traits.api import (
    Any, Bool, Delegate, Enum, Float, Instance, on_trait_change,
)

# Local relative imports
from .enable_traits import bounds_trait, coordinate_trait
from .base import empty_rectangle, intersect_bounds
from .component import Component
from .container import Container
from .canvas import Canvas


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
    stay_inside = Bool(False)

    # Where to anchor vertically on resizes
    vertical_anchor = Enum("bottom", "top", "center")

    # Where to anchor vertically on resizes
    horizontal_anchor = Enum("left", "right", "center")

    # Enable Zoom interaction
    enable_zoom = Bool(False)

    # The zoom tool
    zoom_tool = Instance(ViewportZoomTool)

    # Zoom scaling factor for this viewport - Ratio of old bounds to new bounds
    # Zoom less than 1.0 means we are zoomed out, and more than 1.0 means
    # we are zoomed in.  Zoom should always be positive and nonzero.
    zoom = Float(1.0)

    # Whether to initiate layout on the viewed component.  This is necessary
    # if the component is only viewed through viewports, in which case one
    # of the viewports must lay it out or bounds must be set explicitly
    # on the component.
    initiate_layout = Bool(False)

    min_zoom = Delegate("zoom_tool", modify=True)
    max_zoom = Delegate("zoom_tool", modify=True)

    _component_preferred_size = Any(None)

    # ------------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------------

    def __init__(self, **traits):
        # ensure view_position and view_bounds are set after anchor traits
        view_position = traits.pop("view_position", None)
        view_bounds = traits.pop("view_bounds", None)

        Component.__init__(self, **traits)

        # can't use a default because need bounds to be set first
        if self.component is not None:
            self._initialize_position()
        if view_bounds is not None:
            self.view_bounds = view_bounds
        if view_position is not None:
            self.view_position = view_position

        if "zoom_tool" not in traits:
            self.zoom_tool = ViewportZoomTool(self)
        if self.enable_zoom:
            self._enable_zoom_changed(False, True)

    def components_at(self, x, y, add_containers=False):
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
            damaged_regions = [
                [
                    region[0] - self.view_position[0],
                    region[1] - self.view_position[1],
                    region[2],
                    region[3],
                ]
                for region in damaged_regions
            ]
        super(Viewport, self).invalidate_draw(
            damaged_regions=damaged_regions, self_relative=self_relative
        )

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
            self._component_preferred_size = (
                simple_container_get_preferred_size(
                    self, components=[self.container]
                )
            )
            return self._component_preferred_size
        else:
            return super(Viewport, self).get_preferred_size()

    def viewport_to_component(self, x, y):
        """ Given a coordinate X and Y in the viewport's coordinate system,
        returns and X and Y in the coordinate system of the viewed
        component.
        """
        transform = self.get_event_transform()
        return dot(array([x, y, 1]), transform)[:2]

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
                transform = affine.scale(
                    transform, 1 / self.zoom, 1 / self.zoom
                )
                transform = affine.translate(
                    transform, -self.outer_position[0], -self.outer_position[1]
                )
            else:
                x_offset = self.view_position[0] - self.outer_position[0]
                y_offset = self.view_position[1] - self.outer_position[1]
                transform = affine.translate(transform, x_offset, y_offset)

        return transform

    # ------------------------------------------------------------------------
    # Component interface
    # ------------------------------------------------------------------------

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
                gc.clip_to_rect(
                    x - 0.5, y - 0.5, self.width + 1, self.height + 1
                )

                # There is a two-step transformation from the viewport's
                # "outer" coordinates into the coordinates space of the viewed
                # component: scaling, followed by a translation.
                if self.enable_zoom:
                    if self.zoom != 0:
                        gc.scale_ctm(self.zoom, self.zoom)
                        gc.translate_ctm(
                            x / self.zoom - view_x, y / self.zoom - view_y
                        )
                    else:
                        raise RuntimeError("Viewport zoomed out too far.")
                else:
                    gc.translate_ctm(x - view_x, y - view_y)

                # Now transform the passed-in view_bounds; this is not the same
                # thing as self.view_bounds!
                if view_bounds:
                    # Find the intersection rectangle of the viewport with the
                    # view_bounds, and transform this into the component's
                    # space.
                    clipped_view = intersect_bounds(
                        self.position + self.bounds, view_bounds
                    )
                    if clipped_view != empty_rectangle:
                        # clipped_view and self.position are in the space of
                        # our parent container.  we know that
                        # self.position -> view_x,view_y in the coordinate
                        # space of our component.  So, find the vector from
                        # self.position to clipped_view, then add this to
                        # view_x and view_y to generate the transformed
                        # coordinates of clipped_view in our component's space.
                        offset = array(clipped_view[:2]) - array(self.position)
                        new_bounds = (
                            (offset[0] / self.zoom + view_x),
                            (offset[1] / self.zoom + view_y),
                            clipped_view[2] / self.zoom,
                            clipped_view[3] / self.zoom,
                        )
                        self.component.draw(gc, new_bounds, mode=mode)

    def _do_layout(self):
        if self.initiate_layout:
            self.component.bounds = list(self.component.get_preferred_size())
            self.component.do_layout()

        else:
            super(Viewport, self)._do_layout()

    def _dispatch_stateful_event(self, event, suffix):
        if isinstance(self.component, Component):
            transform = self.get_event_transform(event, suffix)
            event.push_transform(transform, caller=self)
            try:
                self.component.dispatch(event, suffix)
            finally:
                event.pop(caller=self)

    # ------------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------------

    def _enable_zoom_changed(self, old, new):
        """
            Add or remove the zoom tool overlay depending whether
            or not zoom is enabled.
        """
        if self.zoom_tool is None:
            return

        if self.enable_zoom:
            if self.zoom_tool not in self.tools:
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
            self.component.view_bounds = (
                llx,
                lly,
                llx + self.view_bounds[0] - 1,
                lly + self.view_bounds[1] - 1,
            )

    def _component_changed(self, old, new):
        if (old is not None) and (self in old.viewports):
            old.viewports.remove(self)

        if (new is not None) and (self not in new.viewports):
            new.viewports.append(self)
            self._update_component_view_bounds()

    @on_trait_change("component:bounds")
    def _component_bounds_updated(self, obj, name, old, new):
        if name == "bounds":
            delta_x = new[0] - old[0]
            delta_y = new[1] - old[1]
        elif name == "bounds_items":
            delta_x = 0
            delta_y = 0
            if new.index == 1:
                delta_y = new.added[0] - new.removed[0]
            else:
                delta_x = new.added[0] - new.removed[0]
                if len(new.removed) == 2:
                    delta_y = new.added[1] - new.removed[1]
        self._adjust_view_from_component_resize(delta_x, delta_y)

    def _bounds_changed(self, old, new):
        Component._bounds_changed(self, old, new)
        new_w = new[0] / self.zoom
        new_h = new[1] / self.zoom
        w, h = self.view_bounds
        delta_x = new_w - w
        delta_y = new_h - h
        self._adjust_view_from_viewport_resize(delta_x, delta_y)

    def _bounds_items_changed(self, event):
        return self._bounds_changed(None, self.bounds)

    @on_trait_change("view_bounds,view_position")
    def _handle_view_box_changed(self):
        self._update_component_view_bounds()

    def _get_position(self):
        return self.view_position

    def _get_bounds(self):
        return self.view_bounds

    def _initial_position(self):
        x = 0
        y = 0
        if self.vertical_anchor == "top":
            y = self.component.height - self.view_bounds[1]
        elif self.vertical_anchor == "center":
            y = (self.component.height - self.view_bounds[1]) / 2.0
        if self.horizontal_anchor == "right":
            x = self.component.width - self.view_bounds[0]
        elif self.horizontal_anchor == "center":
            x = (self.component.width - self.view_bounds[0]) / 2.0
        return [x, y]

    def _initialize_position(self):
        self.trait_set(view_position=self._initial_position())

    def _adjust_view_from_viewport_resize(self, delta_x, delta_y):
        """ The viewport has been resized, so need to adjust view parameters

        This computes the new view bounds, shifts the view position depending
        on the horizontal and vertical anchoring, and if we are trying to stay
        inside the viewed component, adjusts for that as well.

        Parameters
        ----------
        delta_x : float
            The change in width of the view_bounds in viewed component
            coordinates.

        delta_y : float
            The change in height of the view_bounds in viewed component
            coordinates.
        """
        # resize the view
        w, h = self.view_bounds
        new_w = w + delta_x
        new_h = h + delta_y

        x, y = self._shift_view_position(delta_x, delta_y)

        if self.stay_inside and self.component is not None:
            extra_width = self.component.width - new_w
            extra_height = self.component.height - new_h
            x, y = self._adjust_stay_inside(x, y, extra_width, extra_height)

        self.trait_set(view_bounds=[new_w, new_h], view_position=[x, y])

    def _adjust_view_from_component_resize(self, delta_x, delta_y):
        """ The viewport has been resized, so need to adjust view parameters

        This shifts the view position depending on the horizontal and vertical
        anchoring, and if we are trying to stay inside the viewed component,
        adjusts for that as well.  The view bounds should not change from a
        component resize.

        Parameters
        ----------
        delta_x : float
            The change in width of the viewed component.

        delta_y : float
            The change in height of the viewed component.
        """
        x, y = self._shift_view_position(-delta_x, -delta_y)

        if self.stay_inside and self.component is not None:
            w, h = self.view_bounds
            extra_width = self.component.width - w
            extra_height = self.component.height - h
            x, y = self._adjust_stay_inside(x, y, extra_width, extra_height)

        self.trait_set(view_position=[x, y])

    def _shift_view_position(self, delta_x, delta_y):
        """ Compute new view position, accounting for anchoring

        Parameters
        ----------
        delta_x : float
            The change in width that needs to be accounted for.

        delta_y : float
            The change in height that needs to be accounted for.

        Returns
        -------
        position : tuple of float, float
            The new x, y coordinates for the view position.
        """
        x, y = self.view_position
        if self.vertical_anchor == "top":
            y -= delta_y
        elif self.vertical_anchor == "center":
            y -= delta_y / 2

        if self.horizontal_anchor == "right":
            x -= delta_x
        elif self.horizontal_anchor == "center":
            x -= delta_x / 2
        return x, y

    def _adjust_stay_inside(self, x, y, extra_width, extra_height):
        """ Compute new view position, resisting views outside component

        The algorithm followed is:

        * if the view is smaller than the component, position should be
          between 0 and the amount of extra space
        * otherwise, expand view outside of component based on anchor point

        Parameters
        ----------
        x : float
            The proposed x coordinate of the new view position.

        y : float
            The proposed y coordinate of the new view position.

        extra_width : float
            The difference in width between the viewed component and the view
            bounds (may be negative)

        extra_height : float
            The difference in height between the viewed component and the view
            bounds (may be negative)

        Returns
        -------
        position : tuple of float, float
            The new x, y coordinates for the view position.
        """
        if extra_height >= 0:
            y = min(max(0, y), extra_height)
        elif self.vertical_anchor == "top":
            y = extra_height
        elif self.vertical_anchor == "center":
            y = extra_height / 2
        else:
            y = 0

        if extra_width >= 0:
            x = min(max(0, x), extra_width)
        elif self.horizontal_anchor == "right":
            x = extra_width
        elif self.horizontal_anchor == "center":
            x = extra_width / 2
        else:
            x = 0

        return x, y
