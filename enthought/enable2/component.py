""" Defines the Component class """

# Enthought library imports
from enthought.traits.api import Any, Delegate, Enum, false, Float, Instance, Int, \
                             List, Property, Str, Trait, true

# Local relative imports
from colors import black_color_trait, white_color_trait
from coordinate_box import CoordinateBox
from enable_traits import bounds_trait, coordinate_trait, LineStyle
from interactor import Interactor


coordinate_delegate = Delegate("inner", modify=True)

class Component(CoordinateBox, Interactor):
    """
    Component is the base class for most Enable objects.  In addition to the
    basic position and container features of Component, it also supports
    Viewports and has finite bounds.

    Since Components can have a border and padding, there is an additional set
    of bounds and position attributes that define the "outer box" of the component.
    These cannot be set, since they are secondary attributes (computed from
    the component's "inner" size and margin-area attributes).
    """

    #------------------------------------------------------------------------
    # Object/containment hierarchy traits
    #------------------------------------------------------------------------

    # Our container object
    container = Any    # Instance("Container")

    # A reference to our top-level Enable Window.  This is stored as a shadow
    # attribute if this component is the direct child of the Window; otherwise,
    # the getter function recurses up the containment hierarchy.
    window = Property   # Instance("Window")

    # The list of viewport that are viewing this component
    viewports = List(Instance("enthought.enable2.viewport.Viewport"))


    # A list of strings defining the classes to which this component belongs.
    # These classes will be used to determine how this component is styled,
    # is rendered, is laid out, and receives events.  There is no automatic
    # management of conflicting class names, so if a component is placed
    # into more than one class and that class
    classes = List

    # The optional element ID of this component.
    id = Str("")

    #------------------------------------------------------------------------
    # Layout traits
    #------------------------------------------------------------------------

    #resizable = Enum('h', 'v')

    max_width = Any

    min_width = Any

    max_height = Any

    min_height = Any



    #------------------------------------------------------------------------
    # Padding-related traits
    # Padding in each dimension is defined as the number of pixels that are
    # part of the component but outside of its position and bounds.  Containers
    # need to be aware of padding when doing layout, object collision/overlay
    # calculations, etc.
    #------------------------------------------------------------------------

    # The amount of space to put on the left side of the component
    padding_left = Int(0)

    # The amount of space to put on the right side of the component
    padding_right = Int(0)

    # The amount of space to put on top of the component
    padding_top = Int(0)

    # The amount of space to put below the component
    padding_bottom = Int(0)

    # This property allows a way to set the padding in bulk.  It can either be
    # set to a single Int (which sets padding on all sides) or a tuple/list of
    # 4 Ints representing the left, right, top, bottom padding amounts.  When
    # it is read, this property always returns the padding as a list of 4 elements,
    # even if they are all the same.
    padding = Property

    # Readonly property expressing the total amount of horizontal padding
    hpadding = Property

    # Readonly property expressing the total amount of vertical padding
    vpadding = Property

    # Does the component respond to mouse events occurring over the padding area?
    padding_accepts_focus = true


    #------------------------------------------------------------------------
    # Position and bounds of outer box (encloses the padding and border area)
    # All of these are read-only properties.  To set them directly, use
    # set_outer_coordinates() or set_outer_pos_bounds().
    #------------------------------------------------------------------------

    # The x,y point of the lower left corner of the padding outer box around
    # the component.  Setting this position will move the component, but
    # will not change the padding or bounds.
    # This returns a tuple because modifying the returned value has no effect.
    # To modify outer_position element-wise, use set_outer_position().
    outer_position = Property

    # The number of horizontal and vertical pixels in the padding outer box.
    # Setting these bounds will modify the bounds of the component, but
    # will not change the lower-left position (self.outer_position) or
    # the padding.
    # This returns a tuple because modifying the returned value has no effect.
    # To modify outer_bounds element-wise, use set_outer_bounds().
    outer_bounds = Property

    outer_x = Property
    outer_x2 = Property
    outer_y = Property
    outer_y2 = Property
    outer_width = Property
    outer_height = Property


    #------------------------------------------------------------------------
    # Border and background traits
    #------------------------------------------------------------------------

    # The width of the border around this component.  This is taken into account
    # during layout, but only if the border is visible.
    border_width = Int(1)

    # Is the border visible?  If this is false, then all the other border
    # properties are not
    border_visible = false

    # The line style (i.e. dash pattern) of the border.
    border_dash = LineStyle

    # The color of the border.  Only used if border_visible is True.
    border_color = black_color_trait

    # The background color of this component.  By default all components have
    # a white background.  This can be set to "transparent" or "none" if the
    # component should be see-through.
    bgcolor = white_color_trait


    #------------------------------------------------------------------------
    # Private traits
    #------------------------------------------------------------------------

    # Shadow trait for self.window.  Only gets set if this is the top-level
    # enable component in a Window.
    _window = Any    # Instance("Window")


    #------------------------------------------------------------------------
    # Public methods
    #------------------------------------------------------------------------

    def __init__(self, **traits):
        # The only reason we need the constructor is to make sure our container
        # gets notified of our being added to it.
        if traits.has_key("container"):
            container = traits.pop("container")
            Interactor.__init__(self, **traits)
            container.add(self)
        else:
            Interactor.__init__(self, **traits)
        return

    def draw(self, gc, view_bounds=None, mode="default"):
        """
        Renders this component onto a GraphicsContext.

        "view_bounds" is a 4-tuple (x, y, dx, dy) of the viewed region relative
        to the CTM of the gc.
        """

        # By default, the component is drawn, and then the border is drawn.
        # Subclasses should implement _draw() instead of overriding this
        # method, unless they really know what they are doing.
        self._draw_background(gc, view_bounds, mode)
        self._draw(gc, view_bounds, mode)
        self._draw_border(gc, view_bounds, mode)
        return


    def get_absolute_coords(self, *coords):
        """ Given coordinates relative to this component's origin, returns
        the "absolute" coordinates in the frame of the top-level parent
        Window enclosing this component's ancestor containers.

        Can be called in two ways:
            get_absolute_coords(x, y)
            get_absolute_coords( (x,y) )

        Returns a tuple (x,y) representing the new coordinates.
        """
        if self.container is not None:
            offset_x, offset_y = self.container.get_absolute_coords(*self.position)
        else:
            offset_x, offset_y = self.position
        return (offset_x + coords[0], offset_y + coords[1])

    def request_redraw(self):
        """
        Requests that the component redraw itself.  Usually this means asking
        its parent for a repaint.
        """
        for view in self.viewports:
            view.request_redraw()
        self._request_redraw()
        return

    def is_in(self, x, y):
        # A basic implementation of is_in(); subclasses should provide their
        # own if they are more accurate/faster/shinier.

        if self.padding_accepts_focus:
            bounds = self.outer_bounds
            pos = self.outer_position
        else:
            bounds = self.bounds
            pos = self.position

        return (x >= pos[0]) and (x < pos[0] + bounds[0]) and \
               (y >= pos[1]) and (y < pos[1] + bounds[1])

    def cleanup(self, window):
        """When a window viewing or containing a component is destroyed,
        cleanup is called on the component to give it the opportunity to
        delete any transient state it may have (such as backbuffers)."""
        return


    def set_outer_position(self, ndx, val):
        """
        Since self.outer_position is a property whose value is determined
        by other (primary) attributes, it cannot return a mutable type.
        This method allows generic (i.e. orientation-independent) code
        to set the value of self.outer_position[0] or self.outer_position[1].
        """
        if ndx == 0:
            self.outer_x = val
        else:
            self.outer_y = val
        return

    def set_outer_bounds(self, ndx, val):
        """
        Since self.outer_bounds is a property whose value is determined
        by other (primary) attributes, it cannot return a mutable type.
        This method allows generic (i.e. orientation-independent) code
        to set the value of self.outer_bounds[0] or self.outer_bounds[1].
        """
        if ndx == 0:
            self.outer_width = val
        else:
            self.outer_height = val
        return

    #------------------------------------------------------------------------
    # Protected methods
    #------------------------------------------------------------------------

    def _request_redraw(self):
        if self.container is not None:
            self.container.request_redraw()
        elif self._window:
            self._window.redraw()
        return

    def _draw(self, gc, view_bounds=None, mode="default"):
        # The default Component is an empty object that doesn't do anything
        # when drawn.
        pass

    def _draw_border(self, gc, view_bounds=None, mode="default"):
        """ Utility method to draw the borders around this component """
        if not self.border_visible:
            return

        border_width = self.border_width
        gc.save_state()
        gc.set_line_width(border_width)
        gc.set_line_dash(self.border_dash_)
        gc.set_stroke_color(self.border_color_)
        gc.begin_path()
        gc.rect(self.x - border_width/2.0, self.y - border_width/2.0,
                self.width + 2*border_width - 1, self.height + 2*border_width - 1)
        gc.stroke_path()
        gc.restore_state()
        return

    def _draw_background(self, gc, view_bounds=None, mode="default"):
        if self.bgcolor not in ("transparent", "none"):
            gc.set_fill_color(self.bgcolor_)
            gc.rect(*(self.position + self.bounds))
            gc.fill_path()
        return

    def _get_visible_border(self):
        """ Helper function to return the amount of border, if visible """
        if self.border_visible:
            return self.border_width
        else:
            return 0

    #------------------------------------------------------------------------
    # Event handlers
    #------------------------------------------------------------------------

    def _bounds_changed(self, old, new):
        self.cursor_bounds = new
        if self.container is not None:
            self.container._component_bounds_changed(self)
        return

    def _bounds_items_changed(self, event):
        self.cursor_bounds = self.bounds[:]
        if self.container is not None:
            self.container._component_bounds_changed(self)
        return

    def _container_changed(self, old, new):
        # We don't notify our container of this change b/c the
        # caller who changed our .container should take care of that.
        if new is None:
            self.position = [0,0]
        return

    def _position_changed(self):
        if self.container is not None:
            self.container._component_position_changed(self)
        return

    #------------------------------------------------------------------------
    # Position and padding setters and getters
    #------------------------------------------------------------------------

    def _get_window(self, win):
        return self._window

    def _set_window(self, win):
        self._window = win
        return

    def _get_x(self):
        return self.position[0]

    def _set_x(self, val):
        self.position[0] = val
        return

    def _get_y(self):
        return self.position[1]

    def _set_y(self, val):
        self.position[1] = val
        return

    def _get_padding(self):
        return [self.padding_left, self.padding_right, self.padding_top, self.padding_bottom]

    def _set_padding(self, val):
        old_padding = self.padding

        if type(val) == int:
            self.padding_left = self.padding_right = \
                self.padding_top = self.padding_bottom = val
            self.trait_property_changed("padding", old_padding, [val]*4)
        else:
            # assume padding is some sort of array type
            if len(val) != 4:
                raise RuntimeError, "Padding must be a 4-element sequence type or an int.  Instead, got" + str(val)
            self.padding_left = val[0]
            self.padding_right = val[1]
            self.padding_top = val[2]
            self.padding_bottom = val[3]
            self.trait_property_changed("padding", old_padding, val)
        return

    def _get_hpadding(self):
        return 2*self._get_visible_border() + self.padding_right + self.padding_left

    def _get_vpadding(self):
        return 2*self._get_visible_border() + self.padding_bottom + self.padding_top


    #------------------------------------------------------------------------
    # Outer position setters and getters
    #------------------------------------------------------------------------

    def _get_outer_position(self):
        border = self._get_visible_border()
        pos = self.position
        return (pos[0] - self.padding_left - border,
                pos[1] - self.padding_bottom - border)

    def _set_outer_position(self, new_pos):
        border = self._get_visible_border()
        self.position = [new_pos[0] + self.padding_left + border,
                         new_pos[1] + self.padding_bottom + border]
        return

    def _get_outer_x(self):
        return self.x - self.padding_left - self._get_visible_border()

    def _set_outer_x(self, val):
        self.position[0] = val + self.padding_left + self._get_visible_border()
        return

    def _get_outer_x2(self):
        return self.x2 + self.padding_right + self._get_visible_border()

    def _set_outer_x2(self, val):
        self.x2 = val - self.hpadding
        return

    def _get_outer_y(self):
        return self.y - self.padding_bottom - self._get_visible_border()

    def _set_outer_y(self, val):
        self.position[1] = val + self.padding_bottom + self._get_visible_border()
        return

    def _get_outer_y2(self):
        return self.y2 + self.padding_top + self._get_visible_border()

    def _set_outer_y2(self, val):
        self.y2 = val - self.vpadding
        return

    #------------------------------------------------------------------------
    # Outer bounds setters and getters
    #------------------------------------------------------------------------

    def _get_outer_bounds(self):
        border = self._get_visible_border()
        bounds = self.bounds
        return (bounds[0] + self.hpadding, bounds[1] + self.vpadding)

    def _set_outer_bounds(self, bounds):
        self.bounds = [bounds[0] - self.hpadding, bounds[1] - self.vpadding]
        return

    def _get_outer_width(self):
        return self.outer_bounds[0]

    def _set_outer_width(self, width):
        self.bounds[0] = width - self.hpadding
        return

    def _get_outer_height(self):
        return self.outer_bounds[1]

    def _set_outer_height(self, height):
        self.bounds[1] = height - self.vpadding
        return


# EOF
