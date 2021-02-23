# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
""" Defines the Component class """

from uuid import uuid4

# Enthought library imports
from traits.api import (
    Any, Bool, Delegate, Enum, Float, Instance, Int, List, Property, Str, Trait
)
from kiva.api import FILL, STROKE

# Local relative imports
from .colors import black_color_trait, white_color_trait
from .coordinate_box import CoordinateBox
from .enable_traits import LineStyle, bounds_trait, coordinate_trait
from .interactor import Interactor

coordinate_delegate = Delegate("inner", modify=True)
DEFAULT_DRAWING_ORDER = [
    "background",
    "underlay",
    "mainlayer",
    "border",
    "overlay",
]


class Component(CoordinateBox, Interactor):
    """ Component is the base class for most Enable objects.  In addition to
    the basic position and container features of Component, it also supports
    Viewports and has finite bounds.

    Since Components can have a border and padding, there is an additional set
    of bounds and position attributes that define the "outer box" of the
    components. These cannot be set, since they are secondary attributes
    (computed from the component's "inner" size and margin-area attributes).
    """

    # ------------------------------------------------------------------------
    # Basic appearance traits
    # ------------------------------------------------------------------------

    # Is the component visible?
    visible = Bool(True)

    # Does the component use space in the layout even if it is not visible?
    invisible_layout = Bool(False)

    # Fill the padding area with the background color?
    fill_padding = Bool(False)

    # ------------------------------------------------------------------------
    # Object/containment hierarchy traits
    # ------------------------------------------------------------------------

    # Our container object
    container = Any  # Instance("Container")

    # A reference to our top-level Enable Window.  This is stored as a shadow
    # attribute if this component is the direct child of the Window; otherwise,
    # the getter function recurses up the containment hierarchy.
    window = Property  # Instance("Window")

    # The list of viewport that are viewing this component
    viewports = List(Instance("enable.viewport.Viewport"))

    # ------------------------------------------------------------------------
    # Layout traits
    # ------------------------------------------------------------------------

    # The layout system to use:
    #
    # * 'chaco': Chaco-level layout (the "old" system)
    # * 'enable': Enable-level layout, based on the db/resolver containment
    #   model.
    # NB: this is in preparation for future work
    # layout_switch = Enum("chaco", "enable")

    # Dimensions that this component is resizable in.  For resizable
    # components,  get_preferred_size() is called before their actual
    # bounds are set.
    #
    # * 'v': resizable vertically
    # * 'h': resizable horizontally
    # * 'hv': resizable horizontally and vertically
    # * '': not resizable
    #
    # Note that this setting means only that the *parent* can and should resize
    # this component; it does *not* mean that the component automatically
    # resizes itself.
    resizable = Enum("hv", "h", "v", "")

    # The ratio of the component's width to its height.  This is used by
    # the component itself to maintain bounds when the bounds are changed
    # independently, and is also used by the layout system.
    aspect_ratio = Trait(None, None, Float)

    # When the component's bounds are set to a (width,height) tuple that does
    # not conform to the set aspect ratio, does the component center itself
    # in the free space?
    auto_center = Bool(True)

    # A read-only property that returns True if this component needs layout.
    # It is a reflection of both the value of the component's private
    # _layout_needed attribute as well as any logical layout dependencies with
    # other components.
    layout_needed = Property

    # If the component is resizable, this attribute can be used to specify the
    # amount of space that the component would like to get in each dimension,
    # as a tuple (width, height).  This attribute can be used to establish
    # relative sizes between resizable components in a container: if one
    # component specifies, say, a fixed preferred width of 50 and another one
    # specifies a fixed preferred width of 100, then the latter component will
    # always be twice as wide as the former.
    fixed_preferred_size = Trait(None, None, bounds_trait)

    # ------------------------------------------------------------------------
    # Overlays and underlays
    # ------------------------------------------------------------------------

    # A list of underlays for this plot.  By default, underlays get a chance to
    # draw onto the plot area underneath plot itself but above any images and
    # backgrounds of the plot.
    underlays = List  # [AbstractOverlay]

    # A list of overlays for the plot.  By default, overlays are drawn above
    # the plot and its annotations.
    overlays = List  # [AbstractOverlay]

    # Listen for changes to selection metadata on
    # the underlying data sources, and render them specially?
    use_selection = Bool(False)

    # ------------------------------------------------------------------------
    # Tool and interaction handling traits
    # ------------------------------------------------------------------------

    # An Enable Interactor that all events are deferred to.
    controller = Any

    # Events are *not* automatically considered "handled" if there is a handler
    # defined. Overrides an inherited trait from Enable's Interactor class.
    auto_handle_event = False

    # ------------------------------------------------------------------------
    # Padding-related traits
    # Padding in each dimension is defined as the number of pixels that are
    # part of the component but outside of its position and bounds.  Containers
    # need to be aware of padding when doing layout, object collision/overlay
    # calculations, etc.
    # ------------------------------------------------------------------------

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
    # it is read, this property always returns the padding as a list of four
    # elements even if they are all the same.
    padding = Property

    # Readonly property expressing the total amount of horizontal padding
    hpadding = Property

    # Readonly property expressing the total amount of vertical padding
    vpadding = Property

    # Does the component respond to mouse events over the padding area?
    padding_accepts_focus = Bool(True)

    # ------------------------------------------------------------------------
    # Position and bounds of outer box (encloses the padding and border area)
    # All of these are read-only properties.  To set them directly, use
    # set_outer_coordinates() or set_outer_pos_bounds().
    # ------------------------------------------------------------------------

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

    # ------------------------------------------------------------------------
    # Rendering control traits
    # ------------------------------------------------------------------------

    # The order in which various rendering classes on this component are drawn.
    # Note that if this component is placed in a container, in most cases
    # the container's draw order is used, since the container calls
    # each of its contained components for each rendering pass.
    # Typically, the definitions of the layers are:
    #
    # #. 'background': Background image, shading, and (possibly) borders
    # #. 'mainlayer': The main layer that most objects should draw on
    # #. 'border': A special layer for rendering the border on top of the
    #     component instead of under its main layer (see **overlay_border**)
    # #. 'overlay': Legends, selection regions, and other tool-drawn visual
    #     elements
    draw_order = Instance(list, args=(DEFAULT_DRAWING_ORDER,))

    # If True, then this component draws as a unified whole,
    # and its parent container calls this component's _draw() method when
    # drawing the layer indicated  by **draw_layer**.
    # If False, it tries to cooperate in its container's layer-by-layer
    # drawing. Its parent container calls self._dispatch_draw() with the name
    # of each layer as it goes through its list of layers.
    unified_draw = Bool(False)

    # If **unified_draw** is True for this component, then this attribute
    # determines what layer it will be drawn on.  This is used by containers
    # and external classes, whose drawing loops call this component.
    # If **unified_draw** is False, then this attribute is ignored.
    draw_layer = Str("mainlayer")

    # Draw the border as part of the overlay layer? If False, draw the
    # border as part of the background layer.
    overlay_border = Bool(True)

    # Draw the border inset (on the plot)? If False, draw the border
    # outside the plot area.
    inset_border = Bool(True)

    # ------------------------------------------------------------------------
    # Border and background traits
    # ------------------------------------------------------------------------

    # The width of the border around this component.  This is taken into
    # account during layout, but only if the border is visible.
    border_width = Int(1)

    # Is the border visible?  If this is false, then all the other border
    # properties are not used.
    border_visible = Bool(False)

    # The line style (i.e. dash pattern) of the border.
    border_dash = LineStyle

    # The color of the border.  Only used if border_visible is True.
    border_color = black_color_trait

    # The background color of this component.  By default all components have
    # a white background.  This can be set to "transparent" or "none" if the
    # component should be see-through.
    bgcolor = white_color_trait

    # ------------------------------------------------------------------------
    # Backbuffer traits
    # ------------------------------------------------------------------------

    # Should this component do a backbuffered draw, i.e. render itself to an
    # offscreen buffer that is cached for later use?  If False, then
    # the component will *never* render itself backbuffered, even if asked
    # to do so.
    use_backbuffer = Bool(False)

    # Should the backbuffer extend to the pad area?
    backbuffer_padding = Bool(True)

    # If a draw were to occur, whether the component would actually change.
    # This is useful for determining whether a backbuffer is valid, and is
    # usually set by the component itself or set on the component by calling
    # _invalidate_draw().  It is exposed as a public trait for the rare cases
    # when another component wants to know the validity of this component's
    # backbuffer.
    draw_valid = Bool(False)

    # drawn_outer_position specifies the outer position this component was
    # drawn to on the last draw cycle.  This is used to determine what areas of
    # the screen are damaged.
    drawn_outer_position = coordinate_trait
    # drawn_outer_bounds specifies the bounds of this component on the last
    # draw cycle.  Used in conjunction with outer_position_last_draw
    drawn_outer_bounds = bounds_trait

    # The backbuffer of this component.  In most cases, this is an
    # instance of GraphicsContext, but this requirement is not enforced.
    _backbuffer = Any

    # ------------------------------------------------------------------------
    # New layout/object containment hierarchy traits
    # These are not used yet.
    # ------------------------------------------------------------------------

    # A list of strings defining the classes to which this component belongs.
    # These classes will be used to determine how this component is styled,
    # is rendered, is laid out, and receives events.  There is no automatic
    # management of conflicting class names, so if a component is placed
    # into more than one class and that class
    classes = List

    # The element ID of this component.
    id = Str

    # These will be used by the new layout system, but are currently unused.
    # max_width = Any
    # min_width = Any
    # max_height = Any
    # min_height = Any

    # ------------------------------------------------------------------------
    # Private traits
    # ------------------------------------------------------------------------

    # Shadow trait for self.window.  Only gets set if this is the top-level
    # enable component in a Window.
    _window = Any  # Instance("Window")

    # Whether or not component itself needs to be laid out.  Some times
    # components are composites of others, in which case the layout
    # invalidation relationships should be implemented in layout_needed.
    _layout_needed = Bool(True)

    # ------------------------------------------------------------------------
    # Abstract methods
    # ------------------------------------------------------------------------

    def _do_layout(self):
        """ Called by do_layout() to do an actual layout call; it bypasses some
        additional logic to handle null bounds and setting **_layout_needed**.
        """
        pass

    def _draw_component(self, gc, view_bounds=None, mode="normal"):
        """ Renders the component.

        Subclasses must implement this method to actually render themselves.
        Note: This method is used only by the "old" drawing calls.
        """
        pass

    def _draw_selection(self, gc, view_bounds=None, mode="normal"):
        """ Renders a selected subset of a component's data.

        This method is used by some subclasses. The notion of selection doesn't
        necessarily apply to all subclasses of PlotComponent, but it applies to
        enough of them that it is defined as one of the default draw methods.
        """
        pass

    # ------------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------------

    def __init__(self, **traits):
        # The 'padding' trait sets 4 individual traits in bulk. Make sure that
        # it gets set first before other explicit padding traits get set so
        # they may override the bulk default.
        padding = traits.pop("padding", None)
        padding_traits = {}
        padding_names = [
            "padding_top", "padding_bottom", "padding_left", "padding_right",
        ]
        for name in padding_names:
            try:
                padding_traits[name] = traits.pop(name)
            except KeyError:
                pass

        if "container" in traits:
            # After the component is otherwise configured, make sure our
            # container gets notified of our being added to it.
            container = traits.pop("container")
            super(Component, self).__init__(**traits)
            self._set_padding_traits(padding, padding_traits)
            container.add(self)
        else:
            super(Component, self).__init__(**traits)
            self._set_padding_traits(padding, padding_traits)

    def draw(self, gc, view_bounds=None, mode="default"):
        """ Draws the plot component.

        Parameters
        ----------
        gc : Kiva GraphicsContext
            The graphics context to draw the component on
        view_bounds : 4-tuple of integers
            (x, y, width, height) of the area to draw
        mode : string
            The drawing mode to use; can be one of:

            'normal'
                Normal, antialiased, high-quality rendering
            'overlay'
                The plot component is being rendered over something else,
                so it renders more quickly, and possibly omits rendering
                its background and certain tools
            'interactive'
                The plot component is being asked to render in
                direct response to realtime user interaction, and needs to make
                its best effort to render as fast as possible, even if there is
                an aesthetic cost.
        """
        if self.layout_needed:
            self.do_layout()

        self._draw(gc, view_bounds, mode)

    def draw_select_box(self, gc, position, bounds, width, dash,
                        inset, color, bgcolor, marker_size):
        """ Renders a selection box around the component.

        Subclasses can implement this utility method to render a selection box
        around themselves. To avoid burdening subclasses with various
        selection-box related traits that they might never use, this method
        takes all of its required data as input parameters.

        Parameters
        ----------
        gc : Kiva GraphicsContext
            The graphics context to draw on.
        position : (x, y)
            The position of the selection region.
        bounds : (width, height)
            The size of the selection region.
        width : integer
            The width of the selection box border
        dash : float array
            An array of floating point values specifying the lengths of on and
            off painting pattern for dashed lines.
        inset : integer
            Amount by which the selection box is inset on each side within the
            selection region.
        color : 3-tuple of floats between 0.0 and 1.0
            The R, G, and B values of the selection border color.
        bgcolor : 3-tuple of floats between 0.0 and 1.0
            The R, G, and B values of the selection background.
        marker_size : integer
            Size, in pixels, of "handle" markers on the selection box
        """

        with gc:
            gc.set_line_width(width)
            gc.set_antialias(False)
            x, y = position
            x += inset
            y += inset
            width, height = bounds
            width -= 2 * inset
            height -= 2 * inset
            rect = (x, y, width, height)

            gc.set_stroke_color(bgcolor)
            gc.set_line_dash(None)
            gc.draw_rect(rect, STROKE)

            gc.set_stroke_color(color)
            gc.set_line_dash(dash)
            gc.draw_rect(rect, STROKE)

            if marker_size > 0:
                gc.set_fill_color(bgcolor)
                half_y = y + height / 2.0
                y2 = y + height
                half_x = x + width / 2.0
                x2 = x + width
                marker_positions = (
                    (x, y),
                    (x, half_y),
                    (x, y2),
                    (half_x, y),
                    (half_x, y2),
                    (x2, y),
                    (x2, half_y),
                    (x2, y2),
                )
                gc.set_line_dash(None)
                gc.set_line_width(1.0)
                for pos in marker_positions:
                    gc.rect(
                        pos[0] - marker_size / 2.0,
                        pos[1] - marker_size / 2.0,
                        marker_size,
                        marker_size,
                    )
                gc.draw_path()

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
            offset_x, offset_y = self.container.get_absolute_coords(
                *self.position
            )
        else:
            offset_x, offset_y = self.position
        return (offset_x + coords[0], offset_y + coords[1])

    def get_relative_coords(self, *coords):
        """ Given absolute coordinates (where the origin is the top-left corner
        of the frame in the top-level parent Window) return coordinates
        relative to this component's origin.

        Can be called in two ways:
            get_relative_coords(x, y)
            get_relative_coords( (x,y) )

        Returns a tuple (x,y) representing the new coordinates.
        """
        if self.container is not None:
            offset_x, offset_y = self.container.get_relative_coords(
                *self.position
            )
        else:
            offset_x, offset_y = self.position
        return (coords[0] - offset_x, coords[1] - offset_y)

    def request_redraw(self):
        """
        Requests that the component redraw itself.  Usually this means asking
        its parent for a repaint.
        """
        for view in self.viewports:
            view.request_redraw()

        self._request_redraw()

    def invalidate_draw(self, damaged_regions=None, self_relative=False):
        """ Invalidates any backbuffer that may exist, and notifies our parents
        and viewports of any damaged regions.

        Call this method whenever a component's internal state
        changes such that it must be redrawn on the next draw() call."""
        self.draw_valid = False

        if damaged_regions is None:
            damaged_regions = self._default_damaged_regions()

        if self_relative:
            damaged_regions = [
                [region[0] + self.x, region[1] + self.y, region[2], region[3]]
                for region in damaged_regions
            ]
        for view in self.viewports:
            view.invalidate_draw(
                damaged_regions=damaged_regions,
                self_relative=True,
                view_relative=True,
            )

        if self.container is not None:
            self.container.invalidate_draw(
                damaged_regions=damaged_regions, self_relative=True
            )

        if self._window is not None:
            self._window.invalidate_draw(
                damaged_regions=damaged_regions, self_relative=True
            )

    def invalidate_and_redraw(self):
        """ Convenience method to invalidate our contents and request redraw
        """
        self.invalidate_draw()
        self.request_redraw()

    def is_in(self, x, y):
        # A basic implementation of is_in(); subclasses should provide their
        # own if they are more accurate/faster/shinier.

        if self.padding_accepts_focus:
            bounds = self.outer_bounds
            pos = self.outer_position
        else:
            bounds = self.bounds
            pos = self.position

        return (
            (x >= pos[0])
            and (x < pos[0] + bounds[0])
            and (y >= pos[1])
            and (y < pos[1] + bounds[1])
        )

    def cleanup(self, window):
        """When a window viewing or containing a component is destroyed,
        cleanup is called on the component to give it the opportunity to
        delete any transient state it may have (such as backbuffers)."""

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

    # ------------------------------------------------------------------------
    # Layout-related concrete methods
    # ------------------------------------------------------------------------

    def do_layout(self, size=None, force=False):
        """ Tells this component to do layout at a given size.

        Parameters
        ----------
        size : (width, height)
            Size at which to lay out the component; either or both values can
            be 0. If it is None, then the component lays itself out using
            **bounds**.
        force : Boolean
            Whether to force a layout operation. If False, the component does
            a layout on itself only if **_layout_needed** is True.
            The method always does layout on any underlays or overlays it has,
            even if *force* is False.

        """
        if self.layout_needed or force:
            if size is not None:
                self.bounds = size
            self._do_layout()
            self._layout_needed = False
        for underlay in self.underlays:
            if underlay.visible or underlay.invisible_layout:
                underlay.do_layout()
        for overlay in self.overlays:
            if overlay.visible or overlay.invisible_layout:
                overlay.do_layout()

    def get_preferred_size(self):
        """ Returns the size (width,height) that is preferred for this
        component.

        When called on a component that does not contain other components,
        this method just returns the component bounds.  If the component is
        resizable and can draw into any size, the method returns a size that
        is visually appropriate.  (The component's actual bounds are
        determined by its container's do_layout() method.)
        """
        if self.fixed_preferred_size is not None:
            return self.fixed_preferred_size
        else:
            size = [0, 0]
            outer_bounds = self.outer_bounds
            if "h" not in self.resizable:
                size[0] = outer_bounds[0]
            if "v" not in self.resizable:
                size[1] = outer_bounds[1]
            return size

    # ------------------------------------------------------------------------
    # Protected methods
    # ------------------------------------------------------------------------

    def _set_padding_traits(self, padding, padding_traits):
        """ Set the bulk padding trait and all of the others in the correct
        order.

        Parameters
        ----------
        padding : None, int or list of ints
            The bulk padding.
        padding_traits : dict mapping str to int
            The specific padding traits.
        """
        if padding is not None:
            self.trait_set(padding=padding)
        self.trait_set(**padding_traits)

    def _request_redraw(self):
        if self.container is not None:
            self.container.request_redraw()
        elif self._window:
            self._window.redraw()

    def _default_damaged_regions(self):
        """ Returns the default damaged regions for this Component.

        This consists of the current position/bounds, and the last drawn
        position/bounds
        """
        return [
            list(self.outer_position) + list(self.outer_bounds),
            list(self.drawn_outer_position) + list(self.drawn_outer_bounds),
        ]

    def _draw(self, gc, view_bounds=None, mode="default"):
        """ Draws the component, paying attention to **draw_order**, including
        overlays, underlays, and the like.

        This method is the main draw handling logic in plot components.
        The reason for implementing _draw() instead of overriding the top-level
        draw() method is that the Enable base classes may do things in draw()
        that mustn't be interfered with (e.g., the Viewable mix-in).
        """
        if not self.visible:
            return

        if self.layout_needed:
            self.do_layout()

        self.drawn_outer_position = list(self.outer_position[:])
        self.drawn_outer_bounds = list(self.outer_bounds[:])

        # OpenGL-based graphics-contexts have a `gl_init()` method. We
        # test for this to avoid having to import the OpenGL
        # GraphicsContext just to do an isinstance() check.
        is_gl = hasattr(gc, "gl_init")
        if self.use_backbuffer and (not is_gl):
            if self.backbuffer_padding:
                x, y = self.outer_position
                width, height = self.outer_bounds
            else:
                x, y = self.position
                width, height = self.bounds

            if not self.draw_valid:
                # get a reference to the GraphicsContext class from the object
                GraphicsContext = gc.__class__
                # Some pixels are bigger than others
                pixel_scale = self.window.base_pixel_scale
                size = (int(width * pixel_scale), int(height * pixel_scale))
                if hasattr(GraphicsContext, "create_from_gc"):
                    # For some backends, such as the mac, a much more efficient
                    # backbuffer can be created from the window gc.
                    bb = GraphicsContext.create_from_gc(gc, size)
                else:
                    bb = GraphicsContext(size)

                # Always scale by base_pixel_scale here
                bb.scale_ctm(pixel_scale, pixel_scale)

                # if not fill_padding, then we have to fill the backbuffer
                # with the window color. This is the only way I've found that
                # it works- perhaps if we had better blend support we could set
                # the alpha to 0, but for now doing so causes the backbuffer's
                # background to be white
                if not self.fill_padding:
                    with bb:
                        bb.set_antialias(False)
                        bb.set_fill_color(self.window.bgcolor_)
                        bb.draw_rect((x, y, width, height), FILL)

                # Fixme: should there be a +1 here?
                bb.translate_ctm(-x + 0.5, -y + 0.5)
                # There are a couple of strategies we could use here, but we
                # have to do something about view_bounds.  This is because
                # if we only partially render the object into the backbuffer,
                # we will have problems if we then render with different view
                # bounds.

                for layer in self.draw_order:
                    if layer != "overlay":
                        self._dispatch_draw(layer, bb, view_bounds, mode)

                self._backbuffer = bb
                self.draw_valid = True

            # Blit the backbuffer and then draw the overlay on top
            gc.draw_image(self._backbuffer, (x, y, width, height))
            self._dispatch_draw("overlay", gc, view_bounds, mode)
        else:
            for layer in self.draw_order:
                self._dispatch_draw(layer, gc, view_bounds, mode)

    def _dispatch_draw(self, layer, gc, view_bounds, mode):
        """ Renders the named *layer* of this component.

        This method can be used by container classes that group many components
        together and want them to draw cooperatively. The container iterates
        through its components and asks them to draw only certain layers.
        """
        # Don't render the selection layer if use_selection is false.  This
        # is mostly for backwards compatibility.
        if layer == "selection" and not self.use_selection:
            return
        if self.layout_needed:
            self.do_layout()

        handler = getattr(self, "_draw_" + layer, None)
        if handler:
            handler(gc, view_bounds, mode)

    def _draw_border(self, gc, view_bounds=None, mode="default",
                     force_draw=False):
        """ Utility method to draw the borders around this component

        The *force_draw* parameter forces the method to draw the border; if it
        is false, the border is drawn only when **overlay_border** is True.
        """

        if not self.border_visible:
            return

        if self.overlay_border or force_draw:
            if self.inset_border:
                self._draw_inset_border(gc, view_bounds, mode)
            else:
                border_width = self.border_width
                rect = (
                    self.x - border_width / 2.0,
                    self.y - border_width / 2.0,
                    self.width + 2 * border_width - 1,
                    self.height + 2 * border_width - 1,
                )
                with gc:
                    gc.set_line_width(border_width)
                    gc.set_line_dash(self.border_dash_)
                    gc.set_stroke_color(self.border_color_)
                    gc.draw_rect(rect, STROKE)

    def _draw_inset_border(self, gc, view_bounds=None, mode="default"):
        """ Draws the border of a component.

        Unlike the default Enable border, this one is drawn on the inside of
        the plot instead of around it.
        """
        if not self.border_visible:
            return

        border_width = self.border_width
        rect = (
            self.x + border_width / 2.0 - 0.5,
            self.y + border_width / 2.0 - 0.5,
            self.width - border_width / 2.0,
            self.height - border_width / 2.0,
        )
        with gc:
            gc.set_line_width(border_width)
            gc.set_line_dash(self.border_dash_)
            gc.set_stroke_color(self.border_color_)
            gc.set_antialias(0)
            gc.draw_rect(rect, STROKE)

    # ------------------------------------------------------------------------
    # Protected methods for subclasses to implement
    # ------------------------------------------------------------------------

    def _draw_background(self, gc, view_bounds=None, mode="default"):
        """ Draws the background layer of a component.
        """
        if self.bgcolor not in ("clear", "transparent", "none"):
            if self.fill_padding:
                r = tuple(self.outer_position) + (
                    self.outer_width - 1,
                    self.outer_height - 1,
                )
            else:
                r = tuple(self.position) + (self.width - 1, self.height - 1)

            with gc:
                gc.set_antialias(False)
                gc.set_fill_color(self.bgcolor_)
                gc.draw_rect(r, FILL)

        # Call the enable _draw_border routine
        if not self.overlay_border and self.border_visible:
            # Tell _draw_border to ignore the self.overlay_border
            self._draw_border(gc, view_bounds, mode, force_draw=True)

    def _draw_overlay(self, gc, view_bounds=None, mode="normal"):
        """ Draws the overlay layer of a component.
        """
        for overlay in self.overlays:
            if overlay.visible:
                overlay.overlay(self, gc, view_bounds, mode)

    def _draw_underlay(self, gc, view_bounds=None, mode="normal"):
        """ Draws the underlay layer of a component.
        """
        for underlay in self.underlays:
            # This method call looks funny but it's correct - underlays are
            # just overlays drawn at a different time in the rendering loop.
            if underlay.visible:
                underlay.overlay(self, gc, view_bounds, mode)

    def _get_visible_border(self):
        """ Helper function to return the amount of border, if visible """
        if self.border_visible:
            if self.inset_border:
                return 0
            else:
                return self.border_width
        else:
            return 0

    # ------------------------------------------------------------------------
    # Tool-related methods and event dispatch
    # ------------------------------------------------------------------------

    def dispatch(self, event, suffix):
        """ Dispatches a mouse event based on the current event state.

        Parameters
        ----------
        event : an Enable MouseEvent
            A mouse event.
        suffix : string
            The name of the mouse event as a suffix to the event state name,
            e.g. "_left_down" or "_window_enter".
        """

        # This hasattr check is necessary to ensure compatibility with Chaco
        # components.
        if not getattr(self, "use_draw_order", True):
            self._old_dispatch(event, suffix)
        else:
            self._new_dispatch(event, suffix)

    def _new_dispatch(self, event, suffix):
        """ Dispatches a mouse event

        If the component has a **controller**, the method dispatches the event
        to it, and returns. Otherwise, the following objects get a chance to
        handle the event:

        1. The component's active tool, if any.
        2. Any overlays, in reverse order that they were added and are drawn.
        3. The component itself.
        4. Any underlays, in reverse order that they were added and are drawn.
        5. Any listener tools.

        If any object in this sequence handles the event, the method returns
        without proceeding any further through the sequence. If nothing
        handles the event, the method simply returns.
        """

        # Maintain compatibility with .controller for now
        if self.controller is not None:
            self.controller.dispatch(event, suffix)
            return

        if self._active_tool is not None:
            self._active_tool.dispatch(event, suffix)

        if event.handled:
            return

        # Dispatch to overlays in reverse of draw/added order
        for overlay in self.overlays[::-1]:
            overlay.dispatch(event, suffix)
            if event.handled:
                break

        if not event.handled:
            self._dispatch_stateful_event(event, suffix)

        if not event.handled:
            # Dispatch to underlays in reverse of draw/added order
            for underlay in self.underlays[::-1]:
                underlay.dispatch(event, suffix)
                if event.handled:
                    break

        # Now that everyone who might veto/handle the event has had a chance
        # to receive it, dispatch it to our list of listener tools.
        if not event.handled:
            for tool in self.tools:
                tool.dispatch(event, suffix)

    def _old_dispatch(self, event, suffix):
        """ Dispatches a mouse event.

        If the component has a **controller**, the method dispatches the event
        to it and returns. Otherwise, the following objects get a chance to
        handle the event:

        1. The component's active tool, if any.
        2. Any listener tools.
        3. The component itself.

        If any object in this sequence handles the event, the method returns
        without proceeding any further through the sequence. If nothing
        handles the event, the method simply returns.

        """
        if self.controller is not None:
            self.controller.dispatch(event, suffix)
            return

        if self._active_tool is not None:
            self._active_tool.dispatch(event, suffix)

        if event.handled:
            return

        for tool in self.tools:
            tool.dispatch(event, suffix)
            if event.handled:
                return

        if not event.handled:
            self._dispatch_to_enable(event, suffix)
        return

    def _get_active_tool(self):
        return self._active_tool

    def _set_active_tool(self, tool):
        # Deactivate the existing active tool
        old = self._active_tool
        if old == tool:
            return

        self._active_tool = tool

        if old is not None:
            old.deactivate(self)

        if tool is not None and hasattr(tool, "_activate"):
            tool._activate()

        self.invalidate_and_redraw()

    def _get_layout_needed(self):
        return self._layout_needed

    def _tools_items_changed(self):
        self.invalidate_and_redraw()

    # ------------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------------

    def _id_default(self):
        """ Generate a random UUID for the ID.
        """
        # The first 32bits is plenty.
        return uuid4().hex[:8]

    def _aspect_ratio_changed(self, old, new):
        if new is not None:
            self._enforce_aspect_ratio()

    def _enforce_aspect_ratio(self, notify=True):
        """ This method adjusts the width and/or height of the component so
        that the new width and height match the aspect ratio.  It uses the
        current width and height as a bounding box and finds the largest
        rectangle of the desired aspect ratio that will fit.

        If **notify** is True, then fires trait events for bounds and
        position changing.
        """
        ratio = self.aspect_ratio
        old_w, old_h = self.bounds
        if ratio is None:
            return
        elif ratio == 0:
            self.width = 0
            return
        elif old_h == 0:
            return
        elif int(old_w) == int(ratio * old_h):
            return

        old_aspect = old_w / float(old_h)
        new_pos = None
        if ratio > old_aspect:
            # desired rectangle is wider than bounding box, so use the width
            # and compute a smaller height
            new_w = old_w
            new_h = new_w / ratio
            if self.auto_center:
                new_pos = self.position[:]
                new_pos[1] += (old_h - new_h) / 2.0

        else:
            # desired rectangle is taller than bounding box, so use the height
            # and compute a smaller width
            new_h = old_h
            new_w = new_h * ratio
            if self.auto_center:
                new_pos = self.position[:]
                new_pos[0] += (old_w - new_w) / 2.0

        self.trait_set(bounds=[new_w, new_h], trait_change_notify=notify)
        if new_pos:
            self.trait_set(position=new_pos, trait_change_notify=notify)

    def _bounds_changed(self, old, new):
        self._enforce_aspect_ratio(notify=True)
        if self.container is not None:
            self.container._component_bounds_changed(self)

    def _bounds_items_changed(self, event):
        self._enforce_aspect_ratio(notify=True)
        if self.container is not None:
            self.container._component_bounds_changed(self)

    def _container_changed(self, old, new):
        # We don't notify our container of this change b/c the
        # caller who changed our .container should take care of that.
        if new is None:
            self.position = [0, 0]

    def _position_changed(self, *args):
        if self.container is not None:
            self.container._component_position_changed(self)

    def _position_items_changed(self, *args):
        if self.container is not None:
            self.container._component_position_changed(self)

    def _visible_changed(self, old, new):
        if new:
            self._layout_needed = True

    def _get_window(self):
        if self._window is not None:
            return self._window
        elif self.container is not None:
            return self.container.window
        else:
            return None

    def _set_window(self, win):
        self._window = win

    # ------------------------------------------------------------------------
    # Position and padding setters and getters
    # ------------------------------------------------------------------------

    def _get_x(self):
        return self.position[0]

    def _set_x(self, val):
        self.position[0] = val

    def _get_y(self):
        return self.position[1]

    def _set_y(self, val):
        self.position[1] = val

    def _get_padding(self):
        return [
            self.padding_left,
            self.padding_right,
            self.padding_top,
            self.padding_bottom,
        ]

    def _set_padding(self, val):
        old_padding = self.padding

        if type(val) == int:
            self.padding_left = (
                self.padding_right
            ) = self.padding_top = self.padding_bottom = val
            self.trait_property_changed("padding", old_padding, [val] * 4)
        else:
            # assume padding is some sort of array type
            if len(val) != 4:
                raise RuntimeError(
                    "Padding must be a 4-element sequence "
                    "type or an int.  Instead, got" + str(val)
                )
            self.padding_left = val[0]
            self.padding_right = val[1]
            self.padding_top = val[2]
            self.padding_bottom = val[3]
            self.trait_property_changed("padding", old_padding, val)

    def _get_hpadding(self):
        return (
            2 * self._get_visible_border()
            + self.padding_right
            + self.padding_left
        )

    def _get_vpadding(self):
        return (
            2 * self._get_visible_border()
            + self.padding_bottom
            + self.padding_top
        )

    # ------------------------------------------------------------------------
    # Outer position setters and getters
    # ------------------------------------------------------------------------

    def _get_outer_position(self):
        border = self._get_visible_border()
        pos = self.position
        return (
            pos[0] - self.padding_left - border,
            pos[1] - self.padding_bottom - border,
        )

    def _set_outer_position(self, new_pos):
        border = self._get_visible_border()
        self.position = [
            new_pos[0] + self.padding_left + border,
            new_pos[1] + self.padding_bottom + border,
        ]

    def _get_outer_x(self):
        return self.x - self.padding_left - self._get_visible_border()

    def _set_outer_x(self, val):
        self.position[0] = val + self.padding_left + self._get_visible_border()

    def _get_outer_x2(self):
        return self.x2 + self.padding_right + self._get_visible_border()

    def _set_outer_x2(self, val):
        self.x2 = val - self.padding_right - self._get_visible_border()

    def _get_outer_y(self):
        return self.y - self.padding_bottom - self._get_visible_border()

    def _set_outer_y(self, val):
        self.position[1] = (
            val + self.padding_bottom + self._get_visible_border()
        )

    def _get_outer_y2(self):
        return self.y2 + self.padding_top + self._get_visible_border()

    def _set_outer_y2(self, val):
        self.y2 = val - self.padding_top - self._get_visible_border()

    # ------------------------------------------------------------------------
    # Outer bounds setters and getters
    # ------------------------------------------------------------------------

    def _get_outer_bounds(self):
        bounds = self.bounds
        return (bounds[0] + self.hpadding, bounds[1] + self.vpadding)

    def _set_outer_bounds(self, bounds):
        self.bounds = [bounds[0] - self.hpadding, bounds[1] - self.vpadding]

    def _get_outer_width(self):
        return self.outer_bounds[0]

    def _set_outer_width(self, width):
        self.bounds[0] = width - self.hpadding

    def _get_outer_height(self):
        return self.outer_bounds[1]

    def _set_outer_height(self, height):
        self.bounds[1] = height - self.vpadding
