""" Defines the SimpleZoom class.
"""
from numpy import array 

# Enthought library imports
from enthought.enable2.api import ColorTrait, KeySpec
from enthought.traits.api \
    import Enum, false, true, Float, Instance, Int, List, Str, Trait, true, Tuple

# Enable imports
from enthought.enable2.abstract_overlay import AbstractOverlay
from base_zoom_tool import BaseZoomTool
from tool_history_mixin import ToolHistoryMixin

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
    tool_mode = Enum("range", "box") #Enum("box", "range")
    
    # Is the tool always "on"? If True, left-clicking always initiates
    # a zoom operation; if False, the user must press a key to enter zoom mode.
    always_on = false

    #-------------------------------------------------------------------------
    # Zoom control
    #-------------------------------------------------------------------------

    # The axis to which the selection made by this tool is perpendicular. This
    # only applies in 'range' mode.
    axis = Enum("x", "y")

    #-------------------------------------------------------------------------
    # Interaction control
    #-------------------------------------------------------------------------
    
    # Enable the mousewheel for zooming?
    enable_wheel = true

    # The mouse button that initiates the drag.
    drag_button = Enum("left", "right")
    
    # Conversion ratio from wheel steps to zoom factors.
    wheel_zoom_step = Float(.25)

    # The key press to enter zoom mode, if **always_on** is False.  Has no effect
    # if **always_on** is True.
    enter_zoom_key = Instance(KeySpec, args=("z",))
    
    # The key press to leave zoom mode, if **always_on** is False.  Has no effect
    # if **always_on** is True.
    exit_zoom_key = Instance(KeySpec, args=("z",))

    # Disable the tool after the zoom is completed?
    disable_on_complete = true
    
    # The minimum amount of screen space the user must select in order for
    # the tool to actually take effect.
    minimum_screen_delta = Int(10)
    
    #-------------------------------------------------------------------------
    # Appearance properties (for Box mode)
    #-------------------------------------------------------------------------

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
    
    #------------------------------------------------------------------------
    # Key mappings
    #------------------------------------------------------------------------

    # The key that cancels the zoom and resets the view to the original defaults.
    cancel_zoom_key = Instance(KeySpec, args=("Esc",))

    #------------------------------------------------------------------------
    # Private traits
    #------------------------------------------------------------------------

    # If **always_on** is False, this attribute indicates whether the tool
    # is currently enabled.
    _enabled = false

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
        return

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
        return
    
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
        return

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
    
    def overlay(self, component, gc, view_bounds=None, mode="normal"):
        """ Draws this component overlaid on another component.
        
        Overrides AbstractOverlay.
        """
        if self.event_state == "selecting":
            if self.tool_mode == "range":
                self.overlay_range(component, gc)
            else:
                self.overlay_box(component, gc)
        return

    def overlay_box(self, component, gc):
        """ Draws the overlay as a box.
        """
        if self._screen_start and self._screen_end:
            gc.save_state()
            try:
                gc.set_antialias(0)
                gc.set_line_width(self.border_size)
                gc.set_stroke_color(self.border_color_)
                gc.clip_to_rect(component.x, component.y, component.width, component.height)
                x, y = self._screen_start
                x2, y2 = self._screen_end
                rect = (x, y, x2-x+1, y2-y+1)
                if self.color != "transparent":
                    if self.alpha:
                        color = list(self.color_)
                        if len(color) == 4:
                            color[3] = self.alpha
                        else:
                            color += [self.alpha]
                    else:
                        color = self.color_
                    gc.set_fill_color(color)
                    gc.rect(*rect)
                    gc.draw_path()
                else:
                    gc.rect(*rect)
                    gc.stroke_path()
            finally:
                gc.restore_state()
        return

    def overlay_range(self, component, gc):
        """ Draws the overlay as a range.
        """
        axis_ndx = self._determine_axis()
        lower_left = [0,0]
        upper_right = [0,0]
        lower_left[axis_ndx] = self._screen_start[axis_ndx]
        lower_left[1-axis_ndx] = self.component.position[1-axis_ndx]
        upper_right[axis_ndx] = self._screen_end[axis_ndx] - self._screen_start[axis_ndx]
        upper_right[1-axis_ndx] = self.component.bounds[1-axis_ndx]
        
        gc.save_state()
        try:
            gc.set_antialias(0)
            gc.set_alpha(self.alpha)
            gc.set_fill_color(self.color_)
            gc.set_stroke_color(self.border_color_)
            gc.clip_to_rect(component.x, component.y, component.width, component.height)
            gc.rect(lower_left[0], lower_left[1], upper_right[0], upper_right[1])
            gc.draw_path()
        finally:
            gc.restore_state()
        return
    
    def normal_left_down(self, event):
        """ Handles the left mouse button being pressed while the tool is
        in the 'normal' state.
        
        If the tool is enabled or always on, it starts selecting.
        """
        if self.always_on or self._enabled:
            # we need to make sure that there isn't another active tool that we will
            # interfere with.
            if self.drag_button == "left":
                self._start_select(event)
        return

    def normal_right_down(self, event):
        """ Handles the right mouse button being pressed while the tool is
        in the 'normal' state.
        
        If the tool is enabled or always on, it starts selecting.
        """
        if self.always_on or self._enabled:
            if self.drag_button == "right":
                self._start_select(event)
        return

    def selecting_mouse_move(self, event):
        """ Handles the mouse moving when the tool is in the 'selecting' state.
        
        The selection is extended to the current mouse position.
        """
        self._screen_end = (event.x, event.y)
        self.component.request_redraw()
        event.handled = True
        return
    
    def selecting_left_up(self, event):
        """ Handles the left mouse button being released when the tool is in
        the 'selecting' state.
        
        Finishes selecting and does the zoom.
        """
        if self.drag_button == "left":
            self._end_select(event)
        return

    def selecting_right_up(self, event):
        """ Handles the right mouse button being released when the tool is in 
        the 'selecting' state.
        
        Finishes selecting and does the zoom.
        """
        if self.drag_button == "right":
            self._end_select(event)
        return

    def selecting_mouse_leave(self, event):
        """ Handles the mouse leaving the plot when the tool is in the 
        'selecting' state.
        
        Ends the selection operation without zooming.
        """
        self._end_selecting(event)
        return

    def selecting_key_pressed(self, event):
        """ Handles a key being pressed when the tool is in the 'selecting'
        state.
        
        If the key pressed is the **cancel_zoom_key**, then selecting is 
        canceled.
        """
        if self.cancel_zoom_key.match(event):
            self._end_selecting(event)
            event.handled = True
        return

    def _start_select(self, event):
        """ Starts selecting the zoom region
        """
        if self.component.active_tool in (None, self):
            self.component.active_tool = self
        else:
            self._enabled = False
        self._screen_start = (event.x, event.y)
        self._screen_end = None
        self.event_state = "selecting"
        event.window.set_pointer(self.pointer)
        event.window.set_mouse_owner(self, event.net_transform())
        self.selecting_mouse_move(event)
        return


    def _end_select(self, event):
        """ Ends selection of the zoom region, adds the new zoom range to
        the zoom stack, and does the zoom.
        """
        self._screen_end = (event.x, event.y)

        start = array(self._screen_start)
        end = array(self._screen_end)
        
        if sum(abs(end - start)) < self.minimum_screen_delta:
            self._end_selecting(event)
            event.handled = True
            return
        
        if self.tool_mode == "range":
            axis = self._determine_axis()
            low = self._screen_start[axis]
            high = self._screen_end[axis]
            
            if low > high:
                low, high = high, low
        else:
            low, high = self._screen_start, self._screen_end #self._map_coordinate_box(self._screen_start, self._screen_end)

        new_zoom_range = (low, high)
        self._append_state(new_zoom_range)
        self._do_zoom()
        self._end_selecting(event)
        event.handled = True
        return
    
    def _end_selecting(self, event=None):
        """ Ends selection of zoom region, without zooming.
        """
        if self.disable_on_complete:
            self.disable(event)
        else:
            self.reset()
        self.component.request_redraw()
        if event and event.window.mouse_owner == self:
            event.window.set_mouse_owner(None)
        return

    def _do_zoom(self):
        """ Does the zoom operation.
        """
        # Sets the bounds on the component using _cur_stack_index
        #position, bounds = self._current_state()
        #orig_position, orig_bounds = self._history[0]

        #if self._history_index == 0:
        #    if self.tool_mode == "range":
        #        i = self._get_range_index()
        #        self.component.view_bounds[i] = self._orig_bounds
        #        self.component.view_position[i] = self._orig_position
        #    else:
        #        self.component.view_position = self._orig_position
        #        self.component.view_bounds = self._orig_bounds
                
        #else:    
        #    if self.tool_mode == "range":
        #        if self._zoom_limit_reached(orig_position, orig_bounds, position, bounds):
        #            self._pop_state()
        #            return
        #        i = self._get_range_index()
        #        self.component.view_position[i] = position
        #        self.component.view_bounds[i] = bounds
        #    else:
        #        for ndx in (0, 1):
        #            if self._zoom_limit_reached(orig_position[ndx], orig_bounds[ndx],
        #                                        position[ndx], bounds[ndx]):
                        # pop _current_state off the stack and leave the actual
                        # bounds unmodified.
        #                self._pop_state()
        #                return
        #        self.component.view_position = position
        #        self.component.view_bounds = bounds
            
        #self.component.request_redraw()
        return

    def normal_key_pressed(self, event):
        """ Handles a key being pressed when the tool is in 'normal' state.
        
        If the tool is not always on, this method handles turning it on and
        off when the appropriate keys are pressed. Also handles keys to 
        manipulate the tool history.
        """
        if not self.always_on:
            if not self._enabled and self.enter_zoom_key.match(event):
                if self.component.active_tool in (None, self):
                    self.component.active_tool = self
                    self._enabled = True
                    event.window.set_pointer(self.pointer)
                else:
                    self._enabled = False
                return
            elif self._enabled and self.exit_zoom_key.match(event):
                self._enabled = False
                event.window.set_pointer("arrow")
                return
            
        self._history_handle_key(event)
        
        if event.handled:
            self.component.request_redraw()
        return

    def normal_mouse_wheel(self, event):
        """ Handles the mouse wheel being used when the tool is in the 'normal'
        state.
        
        Scrolling the wheel "up" zooms in; scrolling it "down" zooms out.
        self.component is the viewport
        self.component.component is the canvas

        """
        if self.enable_wheel and event.mouse_wheel != 0:
            position = self.component.view_position
            bounds = self.component.view_bounds
 
            transformed_x = (event.x - position[0]) * self.component.zoom + position[0]
            transformed_y = (event.y - position[1]) * self.component.zoom + position[1]
 
            # Calculate zoom
            if event.mouse_wheel > 0:
                zoom = 1.0 / (1.0 + 0.5 * self.wheel_zoom_step)
                self.component.zoom *= zoom
            elif event.mouse_wheel < 0:
                zoom = 1.0 + 0.5 * self.wheel_zoom_step
                self.component.zoom *= zoom
            
            # Move view to correct position relative to mouse location
            position[0] = transformed_x - (transformed_x - position[0]) / zoom
            position[1] = transformed_y - (transformed_y - position[1]) / zoom
            bounds[0] /= zoom
            bounds[1] /= zoom

            event.handled = True
            self.component.request_redraw()
        return

    def _component_changed(self):
        self._reset_state_to_current()
        return

    #------------------------------------------------------------------------
    # Implementation of PlotComponent interface
    #------------------------------------------------------------------------
    def _activate(self):
        """ Called by PlotComponent to set this as the active tool.
        """
        self.enable()
    
    #------------------------------------------------------------------------
    # implementations of abstract methods on ToolHistoryMixin
    #------------------------------------------------------------------------
    def _reset_state_to_current(self):
        """ Clears the tool history, and sets the current state to be the
        first state in the history.
        """
        if self.tool_mode == "range":
            i = self._get_range_index()
            self._reset_state((self.component.view_position[i],
                               self.component.view_bounds[i]))
        else:
            self._reset_state((self.component.view_position,
                                self.component.view_bounds))

    def _reset_state_pressed(self):
        """ Called when the tool needs to reset its history.  
        
        The history index will have already been set to 0. Implements
        ToolHistoryMixin.
        """
        # First zoom to the set state (ZoomTool handles setting the index=0).
        self._do_zoom()

        # Now reset the state to the current bounds settings.
        self._reset_state_to_current()
        return

    def _prev_state_pressed(self):
        """ Called when the tool needs to advance to the previous state in the
        stack.
        
        The history index will have already been set to the index corresponding
        to the prev state. Implements ToolHistoryMixin.
        """
        self._do_zoom()
        return
    
    def _next_state_pressed(self):
        """ Called when the tool needs to advance to the next state in the stack.
        
        The history index will have already been set to the index corresponding
        to the next state. Implements ToolHistoryMixin.
        """
        self._do_zoom()
        return
    
    ### Persistence ###########################################################

    def __getstate__(self):
        dont_pickle = [
            'always_on',
            'enter_zoom_key',    
            'exit_zoom_key',
            'minimum_screen_delta',
            'event_state',
            'reset_zoom_key',
            'prev_zoom_key',
            'next_zoom_key',
            'pointer',
            '_enabled',
            '_screen_start',
            '_screen_end']
        state = super(SimpleZoom,self).__getstate__()
        for key in dont_pickle:
            if state.has_key(key):
                del state[key]

        return state

# EOF
