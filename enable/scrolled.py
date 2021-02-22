# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!

# Enthought library imports
from traits.api import Any, Bool, DelegatesTo, Float, Instance, Int

# Local, relative imports
from .base import intersect_bounds, empty_rectangle
from .colors import ColorTrait
from .component import Component
from .container import Container
from .viewport import Viewport
from .native_scrollbar import NativeScrollBar


class Scrolled(Container):
    """
    A Scrolled acts like a viewport with scrollbars for positioning the view
    position.  Rather than subclassing from viewport, it delegates to one.
    """

    # The component that we are viewing
    component = Instance(Component)

    # The viewport onto our component
    viewport_component = Instance(Viewport, ())

    # Whether or not the viewport should stay constrained to the bounds
    # of the viewed component
    stay_inside = DelegatesTo("viewport_component")

    # Where to anchor vertically on resizes
    vertical_anchor = DelegatesTo("viewport_component")

    # Where to anchor vertically on resizes
    horizontal_anchor = DelegatesTo("viewport_component")

    # Inside padding is a background drawn area between the edges or scrollbars
    # and the scrolled area/left component.
    inside_padding_width = Int(5)

    # The inside border is a border drawn on the inner edge of the inside
    # padding area to highlight the viewport.
    inside_border_color = ColorTrait("black")
    inside_border_width = Int(0)

    # The background color to use for filling in the padding area.
    bgcolor = ColorTrait("white")

    # Should the horizontal scrollbar be shown?
    horiz_scrollbar = Bool(True)

    # Should the vertical scrollbar be shown?
    vert_scrollbar = Bool(True)

    # Should the scrollbars always be shown?
    always_show_sb = Bool(False)

    # Should the mouse wheel scroll the viewport?
    mousewheel_scroll = Bool(True)

    # Should the viewport update continuously as the scrollbar is dragged,
    # or only when drag terminates (i.e. the user releases the mouse button)
    continuous_drag_update = Bool(True)

    # Override the default value of this inherited trait
    auto_size = False

    # -------------------------------------------------------------------------
    # Traits for support of geophysics plotting
    # -------------------------------------------------------------------------

    # An alternate vertical scroll bar to control this Scrolled, instead of the
    # default one that lives outside the scrolled region.
    alternate_vsb = Instance(Component)

    # The size of the left border space
    leftborder = Float(0)

    # A component to lay out to the left of the viewport area (e.g. a depth
    # scale track)
    leftcomponent = Any

    # -------------------------------------------------------------------------
    # Private traits
    # -------------------------------------------------------------------------

    _vsb = Instance(NativeScrollBar)
    _hsb = Instance(NativeScrollBar)

    # Stores the last horizontal and vertical scroll positions to avoid
    # multiple updates in update_from_viewport()
    _last_hsb_pos = Float(0.0)
    _last_vsb_pos = Float(0.0)

    # Whether or not the viewport region is "locked" from updating via
    # freeze_scroll_bounds()
    _sb_bounds_frozen = Bool(False)

    # Records if the horizontal scroll position has been updated while the
    # Scrolled has been frozen
    _hscroll_position_updated = Bool(False)

    # Records if the vertical scroll position has been updated while the
    # Scrolled has been frozen
    _vscroll_position_updated = Bool(False)

    # Whether or not to the scroll bars should cause an event
    # update to fire on the viewport's view_position.  This is used to
    # prevent redundant events when update_from_viewport() updates the
    # scrollbar position.
    _hsb_generates_events = Bool(True)
    _vsb_generates_events = Bool(True)

    # -------------------------------------------------------------------------
    # Scrolled interface
    # -------------------------------------------------------------------------

    def __init__(self, component, **traits):
        self.component = component
        Container.__init__(self, **traits)
        self._viewport_component_changed()

    def update_bounds(self):
        self._layout_needed = True
        if self._hsb is not None:
            self._hsb._widget_moved = True
        if self._vsb is not None:
            self._vsb._widget_moved = True

    def sb_height(self):
        """ Returns the standard scroll bar height
        """
        # Perhaps a placeholder -- not sure if there's a way to get the
        # standard width or height of a wx scrollbar -- you can set them to
        # whatever you want.
        return 15

    def sb_width(self):
        """ Returns the standard scroll bar width
        """
        return 15

    def freeze_scroll_bounds(self):
        """ Prevents the scroll bounds on the scrollbar from updating until
        unfreeze_scroll_bounds() is called.  This is useful on components with
        view-dependent bounds; when the user is interacting with the scrollbar
        or the viewport, this prevents the scrollbar from resizing underneath
        them.
        """
        if not self.continuous_drag_update:
            self._sb_bounds_frozen = True

    def unfreeze_scroll_bounds(self):
        """ Allows the scroll bounds to be updated by various trait changes.
        See freeze_scroll_bounds().
        """
        self._sb_bounds_frozen = False
        if self._hscroll_position_updated:
            self._handle_horizontal_scroll(self._hsb.scroll_position)
            self._hscroll_position_updated = False
        if self._vscroll_position_updated:
            self._handle_vertical_scroll(self._vsb.scroll_position)
            self._vscroll_position_updated = False
        self.update_from_viewport()
        self.request_redraw()

    # -------------------------------------------------------------------------
    # Trait event handlers
    # -------------------------------------------------------------------------

    def _compute_ranges(self):
        """ Returns the range_x and range_y tuples based on our component
        and our viewport_component's bounds.
        """
        comp = self.component
        viewport = self.viewport_component

        offset = getattr(comp, "bounds_offset", (0, 0))

        ranges = []
        for ndx in (0, 1):
            scrollrange = float(comp.bounds[ndx] - viewport.view_bounds[ndx])
            if round(scrollrange / 20.0) > 0.0:
                ticksize = scrollrange / round(scrollrange / 20.0)
            else:
                ticksize = 1
            ranges.append(
                (
                    offset[ndx],
                    offset[ndx] + comp.bounds[ndx],
                    viewport.view_bounds[ndx],
                    ticksize,
                )
            )

        return ranges

    def update_from_viewport(self):
        """ Repositions the scrollbars based on the current position/bounds of
            viewport_component.
        """
        if self._sb_bounds_frozen:
            return

        x, y = self.viewport_component.view_position
        range_x, range_y = self._compute_ranges()

        modify_hsb = self._hsb and x != self._last_hsb_pos
        modify_vsb = self._vsb and y != self._last_vsb_pos

        if modify_hsb and modify_vsb:
            self._hsb_generates_events = False
        else:
            self._hsb_generates_events = True

        if modify_hsb:
            self._hsb.range = range_x
            self._hsb.scroll_position = x
            self._last_hsb_pos = x

        if modify_vsb:
            self._vsb.range = range_y
            self._vsb.scroll_position = y
            self._last_vsb_pos = y

        if not self._hsb_generates_events:
            self._hsb_generates_events = True

    def _layout_and_draw(self):
        self._layout_needed = True
        self.request_redraw()

    def _component_position_changed(self, component):
        self._layout_needed = True

    def _bounds_changed_for_component(self):
        self._layout_needed = True
        self.update_from_viewport()
        self.request_redraw()

    def _bounds_items_changed_for_component(self):
        self.update_from_viewport()

    def _position_changed_for_component(self):
        self.update_from_viewport()

    def _position_items_changed_for_component(self):
        self.update_from_viewport()

    def _view_bounds_changed_for_viewport_component(self):
        self.update_from_viewport()

    def _view_bounds_items_changed_for_viewport_component(self):
        self.update_from_viewport()

    def _view_position_changed_for_viewport_component(self):
        self.update_from_viewport()

    def _view_position_items_changed_for_viewport_component(self):
        self.update_from_viewport()

    def _component_bounds_items_handler(self, object, event):
        if event.added != event.removed:
            self.update_bounds()

    def _component_bounds_handler(self, object, name, old, new):
        if old is None or new is None or old[0] != new[0] or old[1] != new[1]:
            self.update_bounds()

    def _component_changed(self, old, new):
        if old is not None:
            old.on_trait_change(
                self._component_bounds_handler, "bounds", remove=True
            )
            old.on_trait_change(
                self._component_bounds_items_handler,
                "bounds_items",
                remove=True,
            )
        if new is None:
            self.component = Container()
        else:
            if self.viewport_component:
                self.viewport_component.component = new
            new.container = self
        new.on_trait_change(self._component_bounds_handler, "bounds")
        new.on_trait_change(
            self._component_bounds_items_handler, "bounds_items"
        )
        self._layout_needed = True

    def _bgcolor_changed(self):
        self._layout_and_draw()

    def _inside_border_color_changed(self):
        self._layout_and_draw()

    def _inside_border_width_changed(self):
        self._layout_and_draw()

    def _inside_padding_width_changed(self):
        self._layout_needed = True
        self.request_redraw()

    def _viewport_component_changed(self):
        if self.viewport_component is None:
            self.viewport_component = Viewport(
                stay_inside=self.stay_inside,
                vertical_anchor=self.vertical_anchor,
                horizontal_anchor=self.horizontal_anchor,
            )
        self.viewport_component.view_bounds = self.bounds
        self.viewport_component.component = self.component
        self.viewport_component._initialize_position()
        self.add(self.viewport_component)

    def _alternate_vsb_changed(self, old, new):
        self._component_update(old, new)

    def _leftcomponent_changed(self, old, new):
        self._component_update(old, new)

    def _component_update(self, old, new):
        """ Generic function to manage adding and removing components """
        if old is not None:
            self.remove(old)
        if new is not None:
            self.add(new)

    def _bounds_changed(self, old, new):
        Component._bounds_changed(self, old, new)
        self.update_bounds()

    def _bounds_items_changed(self, event):
        Component._bounds_items_changed(self, event)
        self.update_bounds()

    # -------------------------------------------------------------------------
    # Protected methods
    # -------------------------------------------------------------------------

    def _do_layout(self):
        """ This is explicitly called by _draw().
        """
        self.viewport_component.do_layout()

        # Window is composed of border + scrollbar + canvas in each direction.
        # To compute the overall geometry, first calculate whether component.x
        # + the border fits in the x size of the window.
        # If not, add sb, and decrease the y size of the window by the height
        # of the scrollbar.
        # Now, check whether component.y + the border is greater than the
        # remaining y size of the window.  If it is not, add a scrollbar and
        # decrease the x size of the window by the scrollbar width, and perform
        # the first check again.

        if not self._layout_needed:
            return

        padding = self.inside_padding_width
        scrl_x_size, scrl_y_size = self.bounds
        cont_x_size, cont_y_size = self.component.bounds

        # available_x and available_y are the currently available size for the
        # viewport
        available_x = scrl_x_size - 2 * padding - self.leftborder
        available_y = scrl_y_size - 2 * padding

        # Figure out which scrollbars we will need

        need_x_scrollbar = self.horiz_scrollbar and (
            (available_x < cont_x_size) or self.always_show_sb
        )
        need_y_scrollbar = (
            self.vert_scrollbar
            and ((available_y < cont_y_size) or self.always_show_sb)
        ) or self.alternate_vsb

        if need_x_scrollbar:
            available_y -= self.sb_height()
        if need_y_scrollbar:
            available_x -= self.sb_width()
        if ((available_x < cont_x_size)
                and (not need_x_scrollbar)
                and self.horiz_scrollbar):
            available_y -= self.sb_height()
            need_x_scrollbar = True

        # Put the viewport in the right position
        self.viewport_component.outer_bounds = [available_x, available_y]
        container_y_pos = padding

        if need_x_scrollbar:
            container_y_pos += self.sb_height()
        self.viewport_component.outer_position = [
            padding + self.leftborder,
            container_y_pos,
        ]

        range_x, range_y = self._compute_ranges()

        # Create, destroy, or set the attributes of the horizontal scrollbar
        if need_x_scrollbar:
            bounds = [available_x, self.sb_height()]
            hsb_position = [padding + self.leftborder, 0]
            if not self._hsb:
                self._hsb = NativeScrollBar(
                    orientation="horizontal",
                    bounds=bounds,
                    position=hsb_position,
                    range=range_x,
                    enabled=False,
                )
                v_pos = self.viewport_component.view_position
                self._hsb.scroll_position = v_pos[0]
                self._hsb.on_trait_change(
                    self._handle_horizontal_scroll, "scroll_position"
                )
                self._hsb.on_trait_change(
                    self._mouse_thumb_changed, "mouse_thumb"
                )
                self.add(self._hsb)
            else:
                self._hsb.range = range_x
                self._hsb.bounds = bounds
                self._hsb.position = hsb_position
        elif self._hsb is not None:
            self._hsb = self._release_sb(self._hsb)
        else:
            # We don't need to render the horizontal scrollbar, and we don't
            # have one to update, either.
            pass

        # Create, destroy, or set the attributes of the vertical scrollbar
        if self.alternate_vsb:
            self.alternate_vsb.bounds = [self.sb_width(), available_y]
            self.alternate_vsb.position = [
                2 * padding + available_x + self.leftborder,
                container_y_pos,
            ]

        if need_y_scrollbar and (not self.alternate_vsb):
            bounds = [self.sb_width(), available_y]
            vsb_position = [
                2 * padding + available_x + self.leftborder,
                container_y_pos,
            ]
            if not self._vsb:
                self._vsb = NativeScrollBar(
                    orientation="vertical",
                    bounds=bounds,
                    position=vsb_position,
                    range=range_y,
                )
                v_pos = self.viewport_component.view_position
                self._vsb.scroll_position = v_pos[1]
                self._vsb.on_trait_change(
                    self._handle_vertical_scroll, "scroll_position"
                )
                self._vsb.on_trait_change(
                    self._mouse_thumb_changed, "mouse_thumb"
                )
                self.add(self._vsb)
            else:
                self._vsb.bounds = bounds
                self._vsb.position = vsb_position
                self._vsb.range = range_y
        elif self._vsb:
            self._vsb = self._release_sb(self._vsb)
        else:
            # We don't need to render the vertical scrollbar, and we don't
            # have one to update, either.
            pass

        self._layout_needed = False

    def _release_sb(self, sb):
        if sb is not None:
            if sb == self._vsb:
                sb.on_trait_change(
                    self._handle_vertical_scroll,
                    "scroll_position",
                    remove=True,
                )
            if sb == self._hsb:
                sb.on_trait_change(
                    self._handle_horizontal_scroll,
                    "scroll_position",
                    remove=True,
                )
            self.remove(sb)
            # We shouldn't have to do this, but I'm not sure why the object
            # isn't getting garbage collected.
            # It must be held by another object, but which one?
            sb.destroy()
        return None

    def _handle_horizontal_scroll(self, position):
        if self._sb_bounds_frozen:
            self._hscroll_position_updated = True
            return

        c = self.component
        viewport = self.viewport_component
        offsetx = getattr(c, "bounds_offset", [0, 0])[0]
        if position + viewport.view_bounds[0] <= c.bounds[0] + offsetx:
            if self._hsb_generates_events:
                viewport.view_position[0] = position
            else:
                viewport.trait_setq(
                    view_position=[position, viewport.view_position[1]]
                )

    def _handle_vertical_scroll(self, position):
        if self._sb_bounds_frozen:
            self._vscroll_position_updated = True
            return

        c = self.component
        viewport = self.viewport_component
        offsety = getattr(c, "bounds_offset", [0, 0])[1]
        if position + viewport.view_bounds[1] <= c.bounds[1] + offsety:
            if self._vsb_generates_events:
                viewport.view_position[1] = position
            else:
                viewport.trait_setq(
                    view_position=[viewport.view_position[0], position]
                )

    def _mouse_thumb_changed(self, object, attrname, event):
        if event == "down" and not self.continuous_drag_update:
            self.freeze_scroll_bounds()
        else:
            self.unfreeze_scroll_bounds()

    def _draw(self, gc, view_bounds=None, mode="default"):

        if self.layout_needed:
            self._do_layout()
        with gc:
            self._draw_container(gc, mode)

            self._draw_inside_border(gc, view_bounds, mode)

            dx, dy = self.bounds
            x, y = self.position
            if view_bounds:
                tmp = intersect_bounds((x, y, dx, dy), view_bounds)
                if tmp is empty_rectangle:
                    new_bounds = tmp
                else:
                    new_bounds = (tmp[0] - x, tmp[1] - y, tmp[2], tmp[3])
            else:
                new_bounds = view_bounds

            if new_bounds is not empty_rectangle:
                for component in self.components:
                    if component is not None:
                        with gc:
                            gc.translate_ctm(*self.position)
                            component.draw(gc, new_bounds, mode)

    def _draw_inside_border(self, gc, view_bounds=None, mode="default"):
        width_adjustment = self.inside_border_width / 2
        left_edge = self.x + 1 + self.inside_padding_width - width_adjustment
        right_edge = self.x + self.viewport_component.x2 + 2 + width_adjustment
        bottom_edge = self.viewport_component.y + 1 - width_adjustment
        top_edge = self.viewport_component.y2 + width_adjustment

        with gc:
            gc.set_stroke_color(self.inside_border_color_)
            gc.set_line_width(self.inside_border_width)
            gc.rect(
                left_edge,
                bottom_edge,
                right_edge - left_edge,
                top_edge - bottom_edge,
            )
            gc.stroke_path()

    # -------------------------------------------------------------------------
    # Mouse event handlers
    # -------------------------------------------------------------------------

    def _container_handle_mouse_event(self, event, suffix):
        """
        Implement a container-level dispatch hook that intercepts mousewheel
        events.  (Without this, our components would automatically get handed
        the event.)
        """
        if self.mousewheel_scroll and suffix == "mouse_wheel":
            if self.alternate_vsb:
                self.alternate_vsb._mouse_wheel_changed(event)
            elif self._vsb:
                self._vsb._mouse_wheel_changed(event)
            event.handled = True

    # -------------------------------------------------------------------------
    # Persistence
    # -------------------------------------------------------------------------

    def __getstate__(self):
        state = super(Scrolled, self).__getstate__()
        for key in ["alternate_vsb", "_vsb", "_hsb"]:
            if key in state:
                del state[key]
        return state
