import warnings

from tvtk.api import tvtk
from tvtk import messenger
from traits.api import HasTraits, Any, Callable, Property, Instance, \
        Bool, Enum, Int, on_trait_change

from numpy import arange, zeros, ascontiguousarray, reshape, uint8, any
from enable.api import AbstractWindow, MouseEvent, KeyEvent, \
        CoordinateBox
from enable.graphics_context import ImageGraphicsContextEnable

# Local imports.
from .constants import KEY_MAP


class EnableVTKWindow(AbstractWindow, CoordinateBox):

    # The render window we will be drawing into
    # TODO: Eventually when we move to using the Picker, we will change
    # from observing the RenderWindowInteractor
    control = Instance(tvtk.RenderWindowInteractor)

    # A callable that will be called to request a render. If this
    # is None then we will call render on the control.
    request_render = Callable

    # If events don't get handled by anything in Enable, do they get passed
    # through to the underlying VTK InteractorStyle?
    #
    # This defaults to False because for opaque objects and backgrounds event
    # pass-through can be surprising.  However, it can be useful in the cases
    # when a transparent Enable container or collection of objects overlays
    # a VTK window.
    event_passthrough = Bool(False)

    #------------------------------------------------------------------------
    # Layout traits
    #------------------------------------------------------------------------

    # If we are resizable in a dimension, then when the screen resizes,
    # we will stretch to maintain the amount of empty space between us
    # and the closest border of the screen.
    resizable = Enum("hv", "h", "v", "")

    # The amount of space to put on the left side
    padding_left = Int(0)

    # The amount of space to put on the right side
    padding_right = Int(0)

    # The amount of space to put on top
    padding_top = Int(0)

    # The amount of space to put below
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

    _layout_needed = Bool(False)

    #------------------------------------------------------------------------
    # VTK pipeline objects for rendering and event handling
    #------------------------------------------------------------------------

    # The tvtk.InteractorStyle() object we use
    interactor_style = Any()

    # This is the renderer that we create
    renderer = Any()

    # The VTK ImageData object that hosts the data from our GC's
    # bmp_array.
    _vtk_image_data = Any()

    # The TVTK ImageMapper and Actor2D that render our _gc
    _mapper = Any()
    _actor2d = Any()

    #------------------------------------------------------------------------
    # Private traits for keeping track of mouse state
    #------------------------------------------------------------------------

    # The amount of wheel motion during the previous wheel event
    _wheel_amount = Int(0)

    _left_down = Bool(False)
    _middle_down = Bool(False)
    _right_down = Bool(False)

    #------------------------------------------------------------------------
    # Private traits for managing redraw
    #------------------------------------------------------------------------

    _redraw_needed = Bool(True)

    # Flag to keep from recursing in _vtk_render_event
    _rendering = Bool(False)

    def __init__(self, render_window_interactor, renderer,
            istyle_class=tvtk.InteractorStyle, **traits):
        AbstractWindow.__init__(self, **traits)
        self.control = render_window_interactor
        self.renderer = renderer
        rwi = render_window_interactor

        rwi.interactor_style = istyle_class()

        istyle = rwi.interactor_style
        self._add_observer(istyle, "LeftButtonPressEvent", self._vtk_mouse_button_event)
        self._add_observer(istyle, "LeftButtonReleaseEvent", self._vtk_mouse_button_event)
        self._add_observer(istyle, "MiddleButtonPressEvent", self._vtk_mouse_button_event)
        self._add_observer(istyle, "MiddleButtonReleaseEvent", self._vtk_mouse_button_event)
        self._add_observer(istyle, "RightButtonPressEvent", self._vtk_mouse_button_event)
        self._add_observer(istyle, "RightButtonReleaseEvent", self._vtk_mouse_button_event)
        self._add_observer(istyle, "MouseMoveEvent", self._vtk_mouse_move)
        self._add_observer(istyle, "MouseWheelForwardEvent", self._vtk_mouse_wheel)
        self._add_observer(istyle, "MouseWheelBackwardEvent", self._vtk_mouse_wheel)

        self._add_observer(istyle, "KeyPressEvent", self._on_key_pressed)
        self._add_observer(istyle, "KeyReleaseEvent", self._on_key_released)
        self._add_observer(istyle, "CharEvent", self._on_character)

        # We want _vtk_render_event to be called before rendering, so we
        # observe the StartEvent on the RenderWindow.
        self._add_observer(rwi.render_window, "StartEvent", self._vtk_render_event)
        self._add_observer(istyle, "ExposeEvent", self._vtk_expose_event)
        self.interactor_style = istyle

        self._actor2d = tvtk.Actor2D()
        self.renderer.add_actor(self._actor2d)

        self._mapper = tvtk.ImageMapper()
        self._mapper.color_window = 255
        self._mapper.color_level = 255/2.0
        self._actor2d.mapper = self._mapper

        #self._size = tuple(self._get_control_size())
        self._size = [0,0]
        self._redraw_needed = True
        self._layout_needed = True
        #self._gc = self._create_gc(self._size)

        rwi.initialize()

        #if self.component is not None:
        #    self._paint()

    def _add_observer(self, obj, event, cb):
        """ Adds a vtk observer using messenger to avoid generating uncollectable objects. """
        obj.add_observer(event, messenger.send)
        messenger.connect(tvtk.to_vtk(obj), event, cb)

    def _vtk_render_event(self, vtk_obj, eventname):
        """ Redraw the Enable window, if needed. """
        if not self._rendering:
            self._rendering = True
            try:
                if self._size != self._get_control_size():
                    self._layout_needed = True
                if self._redraw_needed or self._layout_needed:
                    self._paint()
            finally:
                self._rendering = False

    def _vtk_expose_event(self, vtk_obj, eventname):
        #print "Good gods!  A VTK ExposeEvent!"
        pass

    def _pass_event_to_vtk(self, vtk_obj, eventname):
        """ Method to dispatch a particular event name to the appropriate
        method on the vtkInteractorStyle instance
        """
        if "Button" in eventname:
            meth_name = "On"
            if "Left" in eventname:
                meth_name += "LeftButton"
            elif "Right" in eventname:
                meth_name += "RightButton"
            elif "Middle" in eventname:
                meth_name += "MiddleButton"

            if "Press" in eventname:
                meth_name += "Down"
            elif "Release" in eventname:
                meth_name += "Up"

        elif "MouseWheel" in eventname:
            meth_name = "OnMouseWheel"
            if "Forward" in eventname:
                meth_name += "Forward"
            else:
                meth_name += "Backward"

        elif "Key" in eventname:
            meth_name = "OnKey"
            if "Press" in eventname:
                meth_name += "Press"
            else:
                meth_name += "Release"

        elif "Char" in eventname:
            meth_name = "OnChar"

        elif eventname == "MouseMoveEvent":
            meth_name = "OnMouseMove"

        meth = getattr(vtk_obj, meth_name, None)
        if meth is not None:
            meth()
        else:
            warnings.warn("Unable to pass through mouse event '%s' to vtkInteractionStyle" % eventname)
        return

    def _vtk_mouse_button_event(self, vtk_obj, eventname):
        """ Dispatces to self._handle_mouse_event """
        # Check to see if the event falls within the window
        x, y = self.control.event_position
        if not (self.x <= x <= self.x2 and self.y <= y <= self.y2):
            return self._pass_event_to_vtk(vtk_obj, eventname)

        button_map = dict(Left="left", Right="right", Middle="middle")
        action_map = dict(Press="down", Release="up")
        if eventname.startswith("Left"):
            button = "left"
        elif eventname.startswith("Right"):
            button = "right"
        elif eventname.startswith("Middle"):
            button = "middle"
        else:
            # Unable to figure out the appropriate method to dispatch to
            warnings.warn("Unable to create event for", eventname)
            return

        if "Press" in eventname:
            action = "down"
            setattr(self, "_%s_down"%button, True)
        elif "Release" in eventname:
            action = "up"
            setattr(self, "_%s_down"%button, False)
        else:
            # Unable to figure out the appropriate method to dispatch to
            warnings.warn("Unable to create event for", eventname)
            return
        event_name = button + "_" + action
        handled = self._handle_mouse_event(event_name, action)
        if self.event_passthrough and not handled:
            self._pass_event_to_vtk(vtk_obj, eventname)

    def _vtk_mouse_move(self, vtk_obj, eventname):
        x, y = self.control.event_position
        if not (self.x <= x <= self.x2 and self.y <= y <= self.y2):
            return self._pass_event_to_vtk(vtk_obj, eventname)

        handled = self._handle_mouse_event("mouse_move", "move")
        if self.event_passthrough and not handled:
            self._pass_event_to_vtk(vtk_obj, eventname)

    def _vtk_mouse_wheel(self, vtk_obj, eventname):
        x, y = self.control.event_position
        if not (self.x <= x <= self.x2 and self.y <= y <= self.y2):
            return self._pass_event_to_vtk(vtk_obj, eventname)

        if "Forward" in eventname:
            self._wheel_amount = 1
        else:
            self._wheel_amount = -1
        handled = self._handle_mouse_event("mouse_wheel", "wheel")
        if self.event_passthrough and not handled:
            self._pass_event_to_vtk(vtk_obj, eventname)

    def _create_key_event(self, vtk_event, event_type):
        focus_owner = self.focus_owner

        if focus_owner is None:
            focus_owner = self.component

            if focus_owner is None:
                return self._pass_event_to_vtk(vtk_obj, eventname)

        if event_type == 'character':
            key = unicode(self.control.key_sym)
        else:
            key = KEY_MAP.get(self.control.key_sym, None)
            if key is None:
                key = unicode(self.control.key_sym)
            if not key:
                return

        x, y = self.control.event_position

        return KeyEvent(event_type = event_type,
                character=key, x=x, y=y,
                alt_down=bool(self.control.alt_key),
                shift_down=bool(self.control.shift_key),
                control_down=bool(self.control.control_key),
                event=eventname,
                window=self.control)



    def _create_mouse_event(self, event_string):
        """ Returns an enable.MouseEvent that reflects the VTK mouse event.

        **event_string** is just a string: "up", "down", "move", or "wheel".
        It is set in _vtk_mouse_button_event.
        """
        # VTK gives us no event object, so we query the interactor
        # for additional event state.
        rwi = self.control
        x, y = rwi.event_position

        # Offset the event appropriately given our bounds
        x -= self.padding_left
        y -= self.padding_bottom

        wheel = 0
        if event_string in ("up", "down"):
            pass
        elif event_string == "move":
            pass
        elif event_string == "wheel":
            wheel = self._wheel_amount
            # Reset the wheel amount for next time
            self._wheel_amount = 0

        tmp = MouseEvent(x = x, y = y,
                    alt_down = bool(rwi.alt_key),
                    control_down = bool(rwi.control_key),
                    shift_down = bool(rwi.shift_key),
                    left_down = self._left_down,
                    right_down = self._right_down,
                    middle_down = self._middle_down,
                    mouse_wheel = wheel,
                    window = self)
        return tmp

    def _get_control_size(self):
        if self.control is not None:
            return tuple(self.control.size)
        else:
            return (0,0)

    def _redraw(self, coordinates=None):
        " Called by the contained component to request a redraw "
        self._redraw_needed = True
        if self._actor2d is not None:
            self._paint()

    def _on_size(self):
        pass

    def _layout(self, size):
        """ Given a size, set the proper location and bounds of the
        actor and mapper inside our RenderWindow.
        """
        if self.component is None:
            return

        self._size = size
        self.position = [self.padding_left, self.padding_bottom]
        if "h" in self.resizable:
            new_width = size[0] - (self.position[0] + self.padding_right)
            if new_width < 0:
                self.bounds[0] = 0
            else:
                self.bounds[0] = new_width
        if "v" in self.resizable:
            new_height = size[1] - (self.position[1] + self.padding_top)
            if new_height < 0:
                self.bounds[1] = 0
            else:
                self.bounds[1] = new_height

        comp = self.component
        dx, dy = size
        if getattr(comp, "fit_window", False):
            comp.outer_position = [0,0]
            comp.outer_bounds = [dx, dy]
        elif hasattr(comp, "resizable"):
            if "h" in comp.resizable:
                comp.outer_x = 0
                comp.outer_width = dx
            if "v" in comp.resizable:
                comp.outer_y = 0
                comp.outer_height = dy
        comp.do_layout(force=True)

        # Invalidate the GC and the draw flag
        self._gc = None
        self._redraw_needed = True
        self._layout_needed = False

    def _create_gc(self, size, pix_format="rgba32"):
        # Add 1 to each dimension because Kiva uses 0.5 to refer to the center of
        # a pixel.
        width = size[0] + 1
        height = size[1] + 1
        gc = ImageGraphicsContextEnable((width, height), pix_format = pix_format, window=self )
        gc.translate_ctm(0.5, 0.5)
        gc.clear((0,0,0,0))

        imagedata_dimensions = (width, height, 1)
        if self._vtk_image_data is None or any(self._vtk_image_data.dimensions != imagedata_dimensions):
            sz = (width, height, 4)
            img = tvtk.ImageData()
            img.whole_extent = (0, width-1, 0, height-1, 0, 0)
            # note the transposed height and width for VTK (row, column, depth)
            img.dimensions = imagedata_dimensions
            # create a 2d view of the array.  This is a bit superfluous because
            # the GC should be blank at this point in time, but we need to hand
            # the ImageData something.
            try:
                ary = ascontiguousarray(gc.bmp_array[::-1, :, :4])
                ary_2d = reshape(ary, (width * height, 4))
                img.point_data.scalars = ary_2d
            except:
                return self._gc

            if self._vtk_image_data is not None:
                # Just for safety, drop the reference to the previous bmp_array
                # so we don't leak it
                self._vtk_image_data.point_data.scalars = None

            self._actor2d.width = width
            self._actor2d.height = height
            self._mapper.input = img
            self._vtk_image_data = img
        return gc

    def _paint(self, event=None):

        control_size = self._get_control_size()
        size = list(control_size)
        if self._layout_needed or (size != self._size):
            self._layout(size)

        if not self._redraw_needed:
            return

        if self._gc is None:
            self._gc = self._create_gc(self.bounds)

        # Always give the GC a chance to initialize
        self._init_gc()

        # Layout components and draw
        if hasattr(self.component, "do_layout"):
            self.component.do_layout()
        self.component.draw(self._gc, view_bounds=(0, 0, size[0], size[1]))

        # Now transform the image and render it into VTK
        width, height = self.component.outer_bounds
        width += 1
        height += 1
        try:
            ary = ascontiguousarray(self._gc.bmp_array[::-1, :, :4])
            ary_2d = reshape(ary, (width * height, 4))
        except Exception, e:
            warnings.warn("Error reshaping array of shape %s to width and height of (%d, %d)" % (str(ary.shape), width, height))
            return

        # Make sure we paint to the right location on the mapper
        self._vtk_image_data.point_data.scalars = ary_2d
        self._vtk_image_data.modified()
        #self._window_paint(event)
        if self.request_render is not None:
            self.request_render()
        else:
            self.control.render()
        self._redraw_needed = False


    def _set_focus(self):
        #print "set_focus unimplemented"
        pass

    def _capture_mouse(self):
        #print "Capture mouse unimplemented"
        pass

    def _release_mouse(self):
        #print "Release mouse unimplemented"
        pass

    def screen_to_window(self, x, y):
        pass

    def set_pointer(self, pointer):
        pass

    def _set_tooltip(self, tooltip):
        pass


    #------------------------------------------------------------------------
    # Trait property setters/getters
    #------------------------------------------------------------------------

    def _get_padding(self):
        return [self.padding_left, self.padding_right,
                self.padding_top, self.padding_bottom]

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
        return 2*self._get_visible_border() + self.padding_right + \
                self.padding_left

    def _get_vpadding(self):
        return 2*self._get_visible_border() + self.padding_bottom + \
                self.padding_top


    #------------------------------------------------------------------------
    # Trait event handlers
    #------------------------------------------------------------------------

    @on_trait_change("position,position_items")
    def _pos_bounds_changed(self):
        if self._actor2d is not None:
            self._actor2d.position = self.position
        self._redraw_needed

    @on_trait_change("bounds,bounds_items")
    def _bounds_changed(self):
        if self._actor2d is not None:
            self._actor2d.width = self.width
            self._actor2d.height = self.height
        self._redraw_needed = True
        self._layout_needed = True
