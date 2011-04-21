""" Defines the Component class.

FIXME: this appears to be unfinished and unworking as of 2008-08-03.
"""

# Enthought library imports
from traits.api import Any, Bool, Delegate, HasTraits, Instance, \
    Int, List, Property

# Local relative imports
from abstract_component import AbstractComponent
from abstract_layout_controller import AbstractLayoutController
from coordinate_box import CoordinateBox
from render_controllers import AbstractRenderController


coordinate_delegate = Delegate("inner", modify=True)

class Component(CoordinateBox, AbstractComponent):
    """
    Component is the base class for most Enable objects.  In addition to the
    basic position and container features of AbstractComponent, it also supports
    Viewports and has finite bounds.

    Since Components can have a border and padding, there is an additional set
    of bounds and position attributes that define the "outer box" of the component.
    These cannot be set, since they are secondary attributes (computed from
    the component's "inner" size and margin-area attributes).
    """

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
    padding_accepts_focus = Bool(True)

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
    # Public methods
    #------------------------------------------------------------------------

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
    # AbstractComponent interface
    #------------------------------------------------------------------------

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

    #------------------------------------------------------------------------
    # Protected methods
    #------------------------------------------------------------------------

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

    #------------------------------------------------------------------------
    # Padding setters and getters
    #------------------------------------------------------------------------

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




class NewComponent(CoordinateBox, AbstractComponent):

    # A list of strings defining the classes to which this component belongs.
    # These classes will be used to determine how this component is styled,
    # is rendered, is laid out, and receives events.  There is no automatic
    # management of conflicting class names, so if a component is placed
    # into more than one class and that class
    classes = List

    # The optional element ID of this component.


    #------------------------------------------------------------------------
    # Layout traits
    #------------------------------------------------------------------------

    layout_info = Instance(LayoutInfo, args=())

    # backwards-compatible layout properties
    padding_left = Property
    padding_right = Property
    padding_top = Property
    padding_bottom = Property

    padding = Property
    hpadding = Property
    vpadding = Property


    padding_accepts_focus = Bool(True)



class AbstractResolver(HasTraits):
    """
    A Resolver traverses a component DB and matches a specifier.
    """

    def match(self, db, query):
        raise NotImplementedError


class NewContainer(NewComponent):

    # The layout controller determines how the container's internal layout
    # mechanism works.  It can perform the actual layout or defer to an
    # enclosing container's layout controller.  The default controller is
    # a cooperative/recursive layout controller.
    layout_controller = Instance(AbstractLayoutController)


    # The render controller determines how this container and its enclosed
    # components are rendered.  It can actually perform the rendering calls,
    # or defer to an enclosing container's render controller.
    render_controller = Instance(AbstractRenderController)


    resolver = Instance(AbstractResolver)

    # Dict that caches previous lookups.
    _lookup_cache = Any

    def lookup(self, query):
        """
        Returns the component or components matching the given specifier.

        """

