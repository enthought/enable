
# Enthought library imports
from enthought.enable2.traits.rgba_color_trait import RGBAColor
from enthought.traits.api import Instance, true, Int, Any, Float

# Local, relative imports
from base import transparent_color, add_rectangles, intersect_bounds, empty_rectangle
from component import Component
from container import Container
from viewport import Viewport
from native_scrollbar import NativeScrollBar


class Scrolled(Container):

    #---------------------------------------------------------------------------
    #  Trait definitions:
    #---------------------------------------------------------------------------

    component      = Instance(Container)
    viewport_component = Instance(Viewport)
    bgcolor       = RGBAColor("white")
    # Inside padding is a background drawn area between the edges or scrollbars
    # and the scrolled area/left component.
    inside_padding_width = Int(5)
    # The inside border is a border drawn on the inner edge of the inside 
    # padding area to highlight the
    inside_border_color = RGBAColor("black")
    inside_border_width = Int(0)

    horiz_scrollbar = true
    vert_scrollbar = true
    mousewheel_scroll = true # Should the mouse wheel scroll the viewport?
    alternate_vsb = Instance(Component)
    auto_size = False
    leftborder = Float(0) #The size of the left border space
    leftcomponent = Any(None) # None or a component

    #---------------------------------------------------------------------------
    # Private traits
    #---------------------------------------------------------------------------

    _vsb = Instance(NativeScrollBar)
    _hsb = Instance(NativeScrollBar)
    _layout_needed = true


    #---------------------------------------------------------------------------
    # Scrolled interface
    #---------------------------------------------------------------------------

    def __init__(self, component, **traits):
        self.component = component
        Container.__init__( self, **traits )
        self._viewport_component_changed()
        return

    def update_bounds(self):
        self._layout_needed = True
        return

    def sb_height(self):
        """ Returns the standard scroll bar height
        """
        # Perhaps a placeholder -- not sure if there's a way to get the standard
        # width or height of a wx scrollbar -- you can set them to whatever you want.
        return 15

    def sb_width(self):
        """ Returns the standard scroll bar width 
        """
        return 15


    #---------------------------------------------------------------------------
    # Trait event handlers
    #---------------------------------------------------------------------------

    def update_from_viewport(self):
        """ Repositions the scrollbars based on the current position/bounds of
            viewport_component. 
        """

        x, y = self.viewport_component.view_position
        w, h = self.viewport_component.view_bounds
        offsetx, offsety = getattr(self.component, "bounds_offset", (0,0))
        if self._hsb:
            self._hsb.set_position((x + w/2.0 - offsetx)/self.component.bounds[0])
        if self._vsb:
            self._vsb.set_position((y + h/2.0 - offsety)/self.component.bounds[1])
        return


    def _layout_and_draw(self):
        self._layout_needed = True
        self.request_redraw()

    def _bgcolor_changed ( self ):
        self._layout_and_draw()
    def _inside_border_color_changed ( self ):
        self._layout_and_draw()
    def _inside_border_width_changed ( self ):
        self._layout_and_draw()
    def _inside_padding_width_changed(self):
        self._layout_needed = True
        self.request_redraw()

    def _bounds_items_changed_for_component(self):
        self.update_from_viewport()
        return

    def _bounds_changed_for_component(self):
        self.update_from_viewport()
        return

    def _position_changed_for_component(self):
        self.update_from_viewport()
        return

    def _position_items_changed_for_component(self):
        self.update_from_viewport()
        return

    def _view_bounds_changed_for_viewport_component(self):
        self.update_from_viewport()
        return

    def _view_bounds_items_changed_for_viewport_component(self):
        self.update_from_viewport()
        return

    def _view_position_changed_for_viewport_component(self):
        self.update_from_viewport()
        return

    def _view_position_items_changed_for_viewport_component(self):
        self.update_from_viewport()
        return
        
        

    def _component_bounds_items_handler(self, object, new):
        if new.added != new.removed:
            self.update_bounds()

    def _component_bounds_handler(self, object, name, old, new):
        if old == None or new == None or old[0] != new[0] or old[1] != new[1]:
            self.update_bounds()
        return

    def _component_changed ( self, old, new ):
        if old is not None:
            old.on_trait_change(self._component_bounds_handler, 'bounds', remove=True)
            old.on_trait_change(self._component_bounds_items_handler, 'bounds_items', remove=True)
        if new is None:
            self.component = Container()
        else:
            if self.viewport_component:
                self.viewport_component.component = new
        new.on_trait_change(self._component_bounds_handler, 'bounds')
        new.on_trait_change(self._component_bounds_items_handler, 'bounds_items')
        self._layout_needed = True
        return

    def _viewport_component_changed(self):
        if self.viewport_component is None:
            self.viewport_component = Viewport()
        self.viewport_component.component = self.component
        self.viewport_component.view_position = [0,0]
        self.viewport_component.view_bounds = self.bounds
        self.add(self.viewport_component)

    def _alternate_vsb_changed(self, old, new):
        self._component_update(old, new)
        return

    def _leftcomponent_changed(self, old, new):
        self._component_update(old, new)
        return

    def _component_update(self, old, new):
        """Generic function to manage adding and removing
        components"""
        if old is not None:
            self.remove(old)
        if new is not None:
            self.add(new)
        return

    def _bounds_changed ( self, old, new ):
        Component._bounds_changed( self, old, new )
        self.update_bounds()
        return

    def _bounds_items_changed(self, event):
        Component._bounds_items_changed(self, event)
        self.update_bounds()
        return

    def _component_position_changed(self, component):
        self._layout_needed = True
        return


    #---------------------------------------------------------------------------
    # Protected methods
    #---------------------------------------------------------------------------

    def _do_layout ( self ):
        """ This is explicitly called by _draw(). 
        """
        # Window is composed of border + scrollbar + canvas in each direction.
        # To compute the overall geometry, first calculate whether component.x
        # + the border fits in the x size of the window.
        # If not, add sb, and decrease the y size of the window by the height of 
        # the scrollbar.
        # Now, check whether component.y + the border is greater than the remaining
        # y size of the window.  If it is not, add a scrollbar and decrease the x size
        # of the window by the scrollbar width, and perform the first check again.

        if not self._layout_needed:
            return

        padding = self.inside_padding_width
        scrl_x_size, scrl_y_size = self.bounds
        cont_x_size, cont_y_size = self.component.bounds

        # available_x and available_y are the currently available size for the
        # Container
        available_x = scrl_x_size - 2*padding - self.leftborder
        available_y = scrl_y_size - 2*padding

        # Figure out which scrollbars we will need
        need_x_scrollbar = False
        need_y_scrollbar = False
        if available_x < cont_x_size and self.horiz_scrollbar:
            available_y -= self.sb_height()
            need_x_scrollbar = True
        if (available_y < cont_y_size and self.vert_scrollbar) or self.alternate_vsb:
            available_x -= self.sb_width()
            need_y_scrollbar = True
        if (available_x < cont_x_size) and (not need_x_scrollbar) and self.horiz_scrollbar:
            available_y -= self.sb_height()
            need_x_scrollbar = True

        # Put the viewport in the right position
        self.viewport_component.bounds = [available_x, available_y]
        container_y_pos = padding

        if need_x_scrollbar:
            container_y_pos += self.sb_height()
        self.viewport_component.position = [padding + self.leftborder,
                                            container_y_pos]

        # Create, destroy, or set the attributes of the horizontal scrollbar,
        # as necessary
        if need_x_scrollbar:
            bounds = [available_x, self.sb_height()]
            hsb_position = [padding + self.leftborder, 0]
            scrollrange = float(self.component.bounds[0]- \
                                self.viewport_component.bounds[0])
            if round(scrollrange/20.0) > 0.0:
                ticksize = scrollrange/round(scrollrange/20.0)
            else:
                ticksize = 1
            range = (0,self.component.bounds[0],
                     self.viewport_component.bounds[0], ticksize)
            if not self._hsb:
                self._hsb = NativeScrollBar(orientation = 'horizontal',
                                            bounds=bounds,
                                            position=hsb_position,
                                            range=range,
                                            enabled=False
                                            )
                self._hsb.on_trait_change(self._handle_horizontal_scroll,
                                          'scroll_position')
                self.add(self._hsb)
            else: # we already have a scrollbar -- just change the traits we need to
                self._hsb.bounds = bounds
                self._hsb.position = hsb_position
                self._hsb.range = range
        else:
            if self._hsb:
                self._hsb = self._release_sb(self._hsb)
                if hasattr(self.component, "bounds_offset"):
                    pos = self.component.bounds_offset[0]
                else:
                    pos = 0
                self.viewport_component.view_position[0] = pos

        #Create, destroy, or set the attributes of the vertical scrollbar, as necessary
        if self.alternate_vsb:
            self.alternate_vsb.bounds = [self.sb_width(), available_y]
            self.alternate_vsb.position = [2*padding + available_x + self.leftborder,
                                           container_y_pos]

        if need_y_scrollbar and (not self.alternate_vsb):
            bounds = [self.sb_width(), available_y]
            vsb_position = [2*padding + available_x + self.leftborder,
                        container_y_pos]
            #This is to make sure that the line size evenly divides into the
            #scroll range
            scrollrange = float(self.component.bounds[1] \
                                -self.viewport_component.bounds[1])

            if round(scrollrange/20.0) > 0.0:
                ticksize = scrollrange/round(scrollrange/20.0)
            else:
                ticksize = 1

            range = (0,self.component.bounds[1],
                     self.viewport_component.bounds[1], ticksize)
            if not self._vsb:
                self._vsb = NativeScrollBar(orientation = 'vertical',
                                            bounds=bounds,
                                            position=vsb_position,
                                            range=range
                                            )

                self._vsb.on_trait_change(self._handle_vertical_scroll,
                                          'scroll_position')
                self.add(self._vsb)
            else:
                self._vsb.bounds = bounds
                self._vsb.position = vsb_position
                self._vsb.range = range
        else:
            if self._vsb:
                self._vsb = self._release_sb(self._vsb)
                if hasattr(self.component, "bounds_offset"):
                    pos = self.component.bounds_offset[1]
                else:
                    pos = 0
                self.viewport_component.view_position[1] = pos

        self._layout_needed = False
        return

    def _release_sb ( self, sb ):
        if sb is not None:
            if sb == self._vsb:
                sb.on_trait_change( self._handle_vertical_scroll,
                                    'scroll_position', remove = True )
            if sb == self._hsb:
                sb.on_trait_change(self._handle_horizontal_scroll,
                                   'scroll_position', remove=True)
            self.remove(sb)
            # We shouldn't have to do this, but I'm not sure why the object
            # isn't getting garbage collected.
            # It must be held by another object, but which one?
            sb.destroy()
        return None

    def _handle_horizontal_scroll( self, position ):
        if (position + self.viewport_component.view_bounds[0] <=
            self.component.bounds[0]):
            self.viewport_component.view_position[0] = position
        return

    def _handle_vertical_scroll(self, position):
        if (position + self.viewport_component.view_bounds[1] <=
            self.component.bounds[1]):
            self.viewport_component.view_position[1] = position
        return

    def _draw(self, gc, view_bounds=None, mode="default"):

        if self._layout_needed:
            self._do_layout()
        try:
            gc.save_state()
            self._draw_container(gc, mode)

            self._draw_inside_border(gc, view_bounds, mode)

            dx, dy = self.bounds
            x,y = self.position
            if view_bounds:
                tmp = intersect_bounds((x,y,dx,dy), view_bounds)
                if (tmp is empty_rectangle):
                    new_bounds = tmp
                else:
                    new_bounds = (tmp[0]-x, tmp[1]-y, tmp[2], tmp[3])
            else:
                new_bounds = view_bounds

            if new_bounds is not empty_rectangle:
                for component in self.components:
                    if component is not None:
                        try:
                            gc.save_state()
                            gc.translate_ctm(*self.position)
                            component.draw(gc, new_bounds, mode)
                        finally:
                            gc.restore_state()
        finally:
            gc.restore_state()

    def _draw_inside_border(self, gc, view_bounds=None, mode="default"):
        width_adjustment = self.inside_border_width/2
        left_edge = self.x+1 + self.inside_padding_width - width_adjustment
        right_edge = self.x + self.viewport_component.x2+2 + width_adjustment
        bottom_edge = self.viewport_component.y+1 - width_adjustment
        top_edge = self.viewport_component.y2 + width_adjustment

        gc.save_state()
        try:
            gc.set_stroke_color(self.inside_border_color_)
            gc.set_line_width(self.inside_border_width)
            gc.rect(left_edge, bottom_edge,
                    right_edge-left_edge, top_edge-bottom_edge)
            gc.stroke_path()
        finally:
            gc.restore_state()


    #---------------------------------------------------------------------------
    # Mouse event handlers
    #---------------------------------------------------------------------------

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
        return

    #---------------------------------------------------------------------------
    # Persistence
    #---------------------------------------------------------------------------

    #_pickles = ("scale_plot", "selected_tracks")

    def __getstate__(self):
        state = super(Scrolled,self).__getstate__()
        for key in ['alternate_vsb', '_vsb', '_hsb', ]:
            if state.has_key(key):
                del state[key]
        return state

### EOF
