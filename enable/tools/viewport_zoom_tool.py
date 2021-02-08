# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
""" Defines the SimpleZoom class.
"""
from numpy import inf

# Enthought library imports
from traits.api import Bool, Enum, Float, Instance, Int, List, Trait, Tuple

# Enable imports
from enable.base_tool import KeySpec
from enable.colors import ColorTrait
from enable.abstract_overlay import AbstractOverlay

from .base_zoom_tool import BaseZoomTool
from .tool_history_mixin import ToolHistoryMixin


class ViewportZoomTool(AbstractOverlay, ToolHistoryMixin, BaseZoomTool):
    """ Selects a range along the index or value axis.

    The user left-click-drags to select a region to zoom in.
    Certain keyboard keys are mapped to performing zoom actions as well.

    Implements a basic "zoom stack" so the user move go backwards and forwards
    through previous zoom regions.
    """

    # The selection mode:
    #
    # range:
    #   Select a range across a single index or value axis.
    # box:
    #   Perform a "box" selection on two axes.
    tool_mode = Enum("range", "box")  # Enum("box", "range")

    # Is the tool always "on"? If True, left-clicking always initiates
    # a zoom operation; if False, the user must press a key to enter zoom mode.
    always_on = Bool(False)

    # -------------------------------------------------------------------------
    # Zoom control
    # -------------------------------------------------------------------------

    # The axis to which the selection made by this tool is perpendicular. This
    # only applies in 'range' mode.
    axis = Enum("x", "y")

    # -------------------------------------------------------------------------
    # Interaction control
    # -------------------------------------------------------------------------

    # Enable the mousewheel for zooming?
    enable_wheel = Bool(True)

    # The mouse button that initiates the drag.
    drag_button = Enum("left", "right")

    # Conversion ratio from wheel steps to zoom factors.
    wheel_zoom_step = Float(0.25)

    # The key press to enter zoom mode, if **always_on** is False.
    # Has no effect if **always_on** is True.
    enter_zoom_key = Instance(KeySpec, args=("z",))

    # The key press to leave zoom mode, if **always_on** is False.
    # Has no effect if **always_on** is True.
    exit_zoom_key = Instance(KeySpec, args=("z",))

    # Disable the tool after the zoom is completed?
    disable_on_complete = Bool(True)

    # The minimum amount of screen space the user must select in order for
    # the tool to actually take effect.
    minimum_screen_delta = Int(10)

    # The most that this tool will zoom in on the target.  Since zoom is the
    # ratio of the original bounds to the new bounds, a max_zoom value of 2.0
    # would make the tool stop once it had zoomed into a region half the size
    # of the original bounds.
    max_zoom = Float(inf)

    # The most that this tool will zoom out from the target.  For example,
    # a min_zoom of 0.2 would prevent the tool from showing a view zoomed
    # out more than 5 times from the original bounds.
    min_zoom = Float(-inf)

    # -------------------------------------------------------------------------
    # Appearance properties (for Box mode)
    # -------------------------------------------------------------------------

    # The pointer to use when drawing a zoom box.
    pointer = "magnifier"

    # The color of the selection box.
    color = ColorTrait("lightskyblue")

    # The alpha value to apply to **color** when filling in the selection
    # region.  Because it is almost certainly useless to have an opaque zoom
    # rectangle, but it's also extremely useful to be able to use the normal
    # named colors from Enable, this attribute allows the specification of a
    # separate alpha value that replaces the alpha value of **color** at draw
    # time.
    alpha = Trait(0.4, None, Float)

    # The color of the outside selection rectangle.
    border_color = ColorTrait("dodgerblue")

    # The thickness of selection rectangle border.
    border_size = Int(1)

    # The possible event states of this zoom tool.
    event_state = Enum("normal", "selecting")

    # ------------------------------------------------------------------------
    # Key mappings
    # ------------------------------------------------------------------------

    # The key that cancels the zoom and resets the view to the original
    # defaults.
    cancel_zoom_key = Instance(KeySpec, args=("Esc",))

    # ------------------------------------------------------------------------
    # Private traits
    # ------------------------------------------------------------------------

    # If **always_on** is False, this attribute indicates whether the tool
    # is currently enabled.
    _enabled = Bool(False)

    # the original numerical screen ranges
    _orig_position = Trait(None, List, Float)
    _orig_bounds = Trait(None, List, Float)

    # The (x,y) screen point where the mouse went down.
    _screen_start = Trait(None, None, Tuple)

    # The (x,,y) screen point of the last seen mouse move event.
    _screen_end = Trait(None, None, Tuple)

    def __init__(self, component=None, *args, **kw):
        # Support AbstractController-style constructors so that this can be
        # handed in the component it will be overlaying in the constructor
        # without using kwargs.
        self.component = component
        super(ViewportZoomTool, self).__init__(*args, **kw)
        self._reset_state_to_current()

        if self.tool_mode == "range":
            i = self._get_range_index()
            self._orig_position = self.component.view_position[i]
            self._orig_bounds = self.component.view_bounds[i]
        else:
            self._orig_position = self.component.view_position
            self._orig_bounds = self.component.view_bounds

    def enable(self, event=None):
        """ Provides a programmatic way to enable this tool, if
        **always_on** is False.

        Calling this method has the same effect as if the user pressed the
        **enter_zoom_key**.
        """
        if self.component.active_tool != self:
            self.component.active_tool = self
        self._enabled = True
        if event and event.window:
            event.window.set_pointer(self.pointer)

    def disable(self, event=None):
        """ Provides a programmatic way to enable this tool, if **always_on**
        is False.

        Calling this method has the same effect as if the user pressed the
        **exit_zoom_key**.
        """
        self.reset()
        self._enabled = False
        if self.component.active_tool == self:
            self.component.active_tool = None
        if event and event.window:
            event.window.set_pointer("arrow")

    def reset(self, event=None):
        """ Resets the tool to normal state, with no start or end position.
        """
        self.event_state = "normal"
        self._screen_start = None
        self._screen_end = None

    def deactivate(self, component):
        """ Called when this is no longer the active tool.
        """
        # Required as part of the AbstractController interface.
        return self.disable()

    def normal_left_down(self, event):
        """ Handles the left mouse button being pressed while the tool is
        in the 'normal' state.

        If the tool is enabled or always on, it starts selecting.
        """
        if self.always_on or self._enabled:
            # we need to make sure that there isn't another active tool that
            # we will interfere with.
            if self.drag_button == "left":
                self._start_select(event)

    def normal_right_down(self, event):
        """ Handles the right mouse button being pressed while the tool is
        in the 'normal' state.

        If the tool is enabled or always on, it starts selecting.
        """
        if self.always_on or self._enabled:
            if self.drag_button == "right":
                self._start_select(event)

    def normal_mouse_wheel(self, event):
        """ Handles the mouse wheel being used when the tool is in the 'normal'
        state.

        Scrolling the wheel "up" zooms in; scrolling it "down" zooms out.
        self.component is the viewport
        self.component.component is the canvas

        """
        if self.enable_wheel and event.mouse_wheel != 0:

            position = self.component.view_position
            scale = self.component.zoom
            transformed_x = event.x / scale + position[0]
            transformed_y = event.y / scale + position[1]

            # Calculate zoom
            if event.mouse_wheel < 0:
                zoom = 1.0 / (1.0 + 0.5 * self.wheel_zoom_step)
                new_zoom = self.component.zoom * zoom
            elif event.mouse_wheel > 0:
                zoom = 1.0 + 0.5 * self.wheel_zoom_step
                new_zoom = self.component.zoom * zoom

            if new_zoom < self.min_zoom:
                new_zoom = self.min_zoom
                zoom = new_zoom / self.component.zoom
            elif new_zoom > self.max_zoom:
                new_zoom = self.max_zoom
                zoom = new_zoom / self.component.zoom
            self.component.zoom = new_zoom

            x_pos = transformed_x - (transformed_x - position[0]) / zoom
            y_pos = transformed_y - (transformed_y - position[1]) / zoom

            self.component.trait_setq(view_position=[x_pos, y_pos])
            bounds = self.component.view_bounds
            self.component.view_bounds = [bounds[0] / zoom, bounds[1] / zoom]

            event.handled = True
            self.component.request_redraw()

    def _component_changed(self):
        self._reset_state_to_current()

    # ------------------------------------------------------------------------
    # Implementation of PlotComponent interface
    # ------------------------------------------------------------------------
    def _activate(self):
        """ Called by PlotComponent to set this as the active tool.
        """
        self.enable()

    # ------------------------------------------------------------------------
    # implementations of abstract methods on ToolHistoryMixin
    # ------------------------------------------------------------------------
    def _reset_state_to_current(self):
        """ Clears the tool history, and sets the current state to be the
        first state in the history.
        """
        if self.tool_mode == "range":
            i = self._get_range_index()
            self._reset_state(
                (
                    self.component.view_position[i],
                    self.component.view_bounds[i],
                )
            )
        else:
            self._reset_state(
                (self.component.view_position, self.component.view_bounds)
            )
