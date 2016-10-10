"""
Defines the concrete top-level Enable 'Window' class for the Pyglet framework.
Uses the Kiva GL backend.
"""

import warnings

# Pyglet imports
import pyglet
from pyglet import gl, window
from pyglet.window import key

# Enthought library imports
from traits.api import Any, Bool

# Enable imports
from enable.events import MouseEvent, KeyEvent
from enable.graphics_context import GraphicsContextEnable
from enable.abstract_window import AbstractWindow

# local, relative imports
from .constants import ASCII_CONTROL_KEYS, KEY_MAP, \
        POINTER_MAP, TEXT_KEYS


class PygletMouseEvent(object):
    """ Because Pyglet doesn't have a native mouse event object, we use
    this to encapsulate all the possible state when we receive any mouse-
    related event.
    """
    def __init__(self, x, y, dx=0, dy=0, buttons=0, modifiers=None,
                 scroll_x=0, scroll_y=0):
        """ **buttons** is a list of buttons """
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy
        self.modifiers = modifiers
        self.buttons = buttons
        self.scroll_x = scroll_x
        self.scroll_y = scroll_y

        if modifiers is not None:
            self.shift_pressed = bool(modifiers & key.MOD_SHIFT)
            self.ctrl_pressed = bool(modifiers & key.MOD_CTRL)
            self.alt_pressed = bool(modifiers & key.MOD_ALT)
        else:
            self.shift_pressed = self.ctrl_pressed = self.alt_pressed = False
        return


class PygletWindow(window.Window):
    """ Mix-in class that, when combined with a Pyglet event dispatcher of
    some sort, will allow pyglet events to propagate down to Enable.

    The target class should have an attribute named **enable_window**
    that is a reference to the instance of a subclass of AbstractWindow.
    """

    VALID_CTOR_KWARGS = ("width", "height", "caption", "resizable", "style",
                         "fullscreen", "visible", "vsync", "display", "screen",
                         "config", "context")

    def __init__(self, enable_window, **kwargs):
        """ PygletWindow needs a reference to the Enable window; other
        arguments are passed through directly to the pyglet.Window constructor.
        """
        self.enable_window = enable_window

        # This indicates whether or not we should call the Enable window to
        # draw.  If this flag is False, then the draw() method just passes.
        self._dirty = True

        #for key in kwargs:
        #    if key not in PygletWindow.VALID_CTOR_KWARGS:
        #        kwargs.pop(key)
        super(PygletWindow, self).__init__(**kwargs)

        # use a KeyStateHandler to remember the keyboard state.  This
        # is useful since Pyglet separates the notion of keyboard state
        # and character events, and we need to access keyboard state
        # from the on_text handler method.
        self.key_state = key.KeyStateHandler()
        self.push_handlers(self.key_state)

    #-------------------------------------------------------------------------
    # Public methods
    # These are not inherited from/part of the pyglet.window.Window interface
    #-------------------------------------------------------------------------

    def on_draw(self):
        "Called by the mainloop to perform the actual draw"
        if self._dirty:
            self.enable_window._paint()
            self._dirty = False


    def request_redraw(self, coordinates=None):
        """ Called by **self.enable_window** to request a redraw
        **coordinates** is a tuple (x,y,w,h) of a specific sub-region to
        redraw.
        """
        # TODO: Support the **coordinates** argument, perhaps using a direct
        # call to glScissor()
        self._dirty = True

    #-------------------------------------------------------------------------
    # Key/text handling
    #-------------------------------------------------------------------------

    def on_key_press(self, symbol, modifiers):
        # use the bare tuple as the event
        return self._on_key_pressed((symbol, modifiers))

    def on_key_release(self, symbol, modifiers):
        return self._on_key_released((symbol, modifiers))

    def on_text(self, text):
        self._on_character(text)

    def _create_key_event(self, event_type, event):
        if self.enable_window.focus_owner is None:
            focus_owner = self.enable_window.component
        else:
            focus_owner = self.enable_window.focus_owner

        if focus_owner is None:
            return

        if event_type == 'character':
            key = event
            if not key:
                return None
        else:
            key_code = event[0]
            if key_code in KEY_MAP:
                key = KEY_MAP.get(key_code)
            else:
                key = key_code

        return KeyEvent(
            event_type = event_type,
            character = key,
            alt_down = keys[key.LALT] | keys[key.RALT],
            control_down = keys[key.LCTRL] | keys[key.RCTRL],
            shift_down = keys[key.LSHIFT] | keys[key.RSHIFT],
            x = self._mouse_x,
            y = self._mouse_y,
            window = self.enable_window)

    def on_text_motion(self, motion):
        # TODO: See notes.
        pass

    def on_text_motion_select(self, motion):
        # TODO: See notes.
        pass

    #-------------------------------------------------------------------------
    # Mouse handling
    #-------------------------------------------------------------------------

    def on_mouse_motion(self, x, y, dx, dy):
        event = PygletMouseEvent(x, y, dx, dy)
        self.enable_window._handle_mouse_event("mouse_move", event, set_focus=False)

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        # TODO: Determine the difference between this and on_mouse_motion;
        # confirm that the correct buttons in **buttons** are down.
        event = PygletMouseEvent(x, y, dx, dy, buttons=buttons, modifiers=modifiers)
        self.enable_window._handle_mouse_event("mouse_move", event, set_focus=False)

    def on_mouse_press(self, x, y, button, modifiers):
        return self._on_mouse_updown(x, y, button, modifiers, "down")

    def on_mouse_release(self, x, y, button, modifiers):
        return self._on_mouse_updown(x, y, button, modifiers, "up")

    def _on_mouse_updown(self, x, y, button, modifiers, which="down"):
        event = PygletMouseEvent(x, y, buttons=button, modifiers=modifiers)
        mouse = pyglet.window.mouse
        if button == mouse.LEFT:
            name = "left"
        elif button == mouse.MIDDLE:
            name = "middle"
        elif button == mouse.RIGHT:
            name = "right"
        else:
            raise RuntimeError("Unknown mouse button state in _on_mouse_updown()")
        self.enable_window._handle_mouse_event(name+"_"+which, event, set_focus=False)
        # TODO: Confirm that we should consume mouse press/release events
        return True

    def on_mouse_scroll(self, x, y, scroll_x, scroll_y):
        # TODO: Handle scroll_x
        event = PygletMouseEvent(x, y, scroll_x=scroll_x, scroll_y=scroll_y)
        self.enable_window._handle_mouse_event("mouse_wheel", event, set_focus=False)

    def on_mouse_enter(self, x, y):
        event = PygletMouseEvent(x, y)
        self.enable_window._handle_mouse_event("mouse_enter", event, set_focus=False)

    def on_mouse_leave(self, x, y):
        event = PygletMouseEvent(x, y)
        self.enable_window._handle_mouse_event("mouse_leave", event, set_focus=False)

    #-------------------------------------------------------------------------
    # Window
    #-------------------------------------------------------------------------

    def on_resize(self, width, height):
        self._dirty = True
        self.enable_window.resized = (width, height)

    def on_close(self):
        pass

    def on_expose(self):
        pass

    def on_move(self, x, y):
        """The window was moved.  x is the distance from the left edge of the
        screen to the left edge of the window.  y is the distance from the top
        edge of the screen to the top edge of the window.
        """
        pass

    def on_activate(self):
        """ The window was activated. """
        self._dirty = True

    def on_deactivate(self):
        """ The window lost focus. """
        pass

    def on_show(self):
        """ The window was shown. """
        self._dirty = True

    def on_hide(self):
        """ The window was minimized or hidden. """
        pass

    #-------------------------------------------------------------------------
    # GL context stuff - see the pyglet.window.Window documentation on these
    # methods
    #-------------------------------------------------------------------------

    def on_context_lost(self):
        pass

    def on_context_state_lost(self):
        pass

class Window(AbstractWindow):

    # If this is True, then the screen is configured for full-screen
    # antialiasing.  This can be noticeably slower, however.
    enable_antialias = Bool(True)

    _cursor_color = Any  # PZW: figure out the correct type for this...

    # This is set by downstream components to notify us of whether or not
    # the current drag operation should return DragCopy, DragMove, or DragNone.
    _drag_result = Any

    def __init__(self, parent=None, id=-1, pos=None, size=None, config=None,
        fullscreen=False, resizable=True, vsync=True, **traits):
        """ **parent** is an unneeded argument with the pyglet backend, but
        we need to preserve compatibility with other AbstractWindow
        subclasses.
        """
        # TODO: Fix fact that other backends' Window classes use positional
        # arguments

        self.control = None
        AbstractWindow.__init__(self, **traits)
        self._mouse_captured = False

        # Due to wx wonkiness, we don't reliably get cursor position from
        # a wx KeyEvent.  Thus, we manually keep track of when we last saw
        # the mouse and use that information instead.  These coordinates are
        # in the wx coordinate space, i.e. pre-self._flip_y().
        self._last_mouse_pos = (0, 0)

        # Try to get antialiasing, both for quality rendering and for
        # reproducible results. For example, line widths are measured in the
        # X or Y directions rather than perpendicular to the line unless if
        # antialiasing is enabled.
        display = window.get_platform().get_default_display()
        screen = display.get_default_screen()
        if config is None:
            if self.enable_antialias:
                template_config = gl.Config(double_buffer=True, sample_buffers=True,
                    samples=4)
            else:
                template_config = gl.Config(double_buffer=False)
            try:
                config = screen.get_best_config(template_config)
            except window.NoSuchConfigException:
                # Rats. No antialiasing.
                config = screen.get_best_config(gl.Config(double_buffer=True))
        # Create the underlying control.
        kwds = dict(config=config, fullscreen=fullscreen,
            resizable=resizable, vsync=vsync)
        if size is not None and not fullscreen:
            kwds['width'], kwds['height'] = size
        self.control = PygletWindow(enable_window=self, **kwds)
        if pos is not None:
            self.control.set_location(*pos)

        return

    def _flip_y(self, y):
        """ Convert from a Kiva to a Pyglet y-coordinate.
        Since pyglet uses the same convention as Kiva, this is a no-op.
        """
        return y

    def _on_erase_background(self, event):
        pass

    def _resized_changed(self, event):
        self._size = (self.control.width, self.control.height)
        width, height = self._size
        component = self.component
        if hasattr(component, "fit_window") and component.fit_window:
            component.outer_position = [0,0]
            component.outer_bounds = [width, height]
        elif hasattr(component, "resizable"):
            if "h" in component.resizable:
                component.outer_x = 0
                component.outer_width = width
            if "v" in component.resizable:
                component.outer_y = 0
                component.outer_height = height
        return

    def _capture_mouse(self):
        "Capture all future mouse events"
        # TODO: Figure out how to do mouse capture.
        # Pyglet's Window class has a set_mouse_exclusive() mode, but this
        # makes the cursur invisible as well.  It really is more of a
        # full-screen "Game Mode", and not designed for mouse capture in a
        # traditional GUI toolkit sense.

        #if not self._mouse_captured:
        #    self.control.set_mouse_exclusive(True)
        #    self._mouse_captured = True
        pass

    def _release_mouse(self):
        "Release the mouse capture"
        #if self._mouse_captured:
        #    self._mouse_captured = False
        #    self.control.set_mouse_exclusive(False)
        pass

    def _create_mouse_event(self, event):
        """ Convert a Pyglet mouse event into an Enable MouseEvent.

        Since Pyglet doesn't actually have a mouse event object like WX or Qt,
        PygletWindow actually does most of the work of creating an Enable
        MouseEvent when various things happen, and calls
        AbstractWindow._handle_mouse_event with that object.
        _handle_mouse_event() then calls this method with that object.

        AbstractWindow._on_window_leave() also calls this method.
        """
        if event is not None:
            x = event.x
            y = event.y
            self._last_mouse_pos = (x, y)
            mouse = pyglet.window.mouse
            buttons = event.buttons
            if buttons is None:
                buttons = 0
            return MouseEvent( x = x, y = y,
                               alt_down     = event.alt_pressed,
                               control_down = event.ctrl_pressed,
                               shift_down   = event.shift_pressed,
                               left_down    = bool(mouse.LEFT & buttons),
                               middle_down  = bool(mouse.MIDDLE & buttons),
                               right_down   = bool(mouse.RIGHT & buttons),
                               mouse_wheel  = event.scroll_y,
                               window = self)
        else:
            # If no event specified, make one up:
            x = self.control._mouse_x
            y = self.control._mouse_y
            self._last_mouse_pos = (x, y)
            return MouseEvent( x = x, y = y,
                               alt_down     = event.alt_pressed,
                               control_down = event.ctrl_pressed,
                               shift_down   = event.shift_pressed,
                               left_down    = False,
                               middle_down  = False,
                               right_down   = False,
                               mouse_wheel  = 0,
                               window = self)

    def _create_gc(self, size, pix_format = "rgba32"):
        "Create a Kiva graphics context of a specified size."
        # Unlike the vector-based Agg and Quartz GraphicsContexts which place
        # pixel coordinates at the lower-left corner, the Pyglet backend is
        # raster-based and places coordinates at the center of pixels.
        gc = GraphicsContextEnable((size[0]+1, size[1]+1), window=self)
        gc.translate_ctm(0.5, 0.5)
        gc.gl_init()
        return gc

    def _init_gc(self):
        #gc = self._gc
        #gc.gl_init()
        pass

    def _redraw(self, coordinates=None):
        "Request a redraw of the window"
        if self.control is not None:
            self.control.request_redraw(coordinates)

    def _get_control_size(self):
        "Get the size of the underlying toolkit control"
        if self.control is not None:
            return (self.control.width, self.control.height)
        else:
            return None

    def set_pointer(self, pointer):
        "Set the current pointer (i.e. cursor) shape"
        if pointer == "blank":
            self.control.set_mouse_visible(False)
        elif pointer in POINTER_MAP:
            self.control.set_mouse_visible(True)
            cursor = self.control.get_system_mouse_cursor(POINTER_MAP[pointer])
            self.control.set_mouse_cursor(cursor)
        else:
            warnings.warn("Unable to set mouse pointer '%s' in"
                          "Enable's Pyglet backend." % pointer)
            cursor = self.control.get_system_mouse_cursor(POINTER_MAP["arrow"])
            self.control.set_mouse_cursor(cursor)
        return

    def set_timer_interval(self, component, interval):
        """ Set up or cancel a timer for a specified component.  To cancel the
        timer, set interval=None.
        """
        raise NotImplementedError("set_timer_interval() not implemented yet in Pyglet backend.")

    def _set_focus(self):
        """ Sets the keyboard focus to this window.

        Since Pyglet is not a windowing system, there are not other windows we
        might lose focus to; the entire application has focus or it doesn't.
        This attempts to make the application regain focus.
        """
        self.control.activate()

    #-------------------------------------------------------------------------
    # Unnecessary methods but provided for compatibility
    #-------------------------------------------------------------------------
    def _paint(self, event=None):
        # Override the base class _paint() method because we need to call
        # _create_gc() each time *before* self.component draws.

        size = self._get_control_size()
        self._size = tuple(size)
        self._gc = self._create_gc(size)
        self.control.clear()
        gc = self._gc
        if hasattr(self.component, "do_layout"):
            self.component.do_layout()
        gc.clear(self.bgcolor_)
        self.component.draw(gc, view_bounds=(0, 0, size[0], size[1]))
        self._update_region = []
        #self.control.flip()
        return

    def _window_paint(self, event):
        "Do a backend-specific screen update"
        # We don't actually have to do anything here, and our implementation
        # of _paint() doesn't even call this method.
        #
        # In other backends where the self.component.draw(gc) call just renders
        # onto an in-screen GraphicsContext, this method is used to do a
        # platform-specific blit.  In the case of Pyglet, the component.draw()
        # method executes immediately on the current OpenGL context, so there
        # is no additional step needed here.
        pass

    def screen_to_window(self, x, y):
        """ Convert screen coordinates with the origin at the upper-left-hand
        corner to local pyglet window coordinates.
        """
        x0, y0 = self.control.get_location()
        x -= x0
        y -= y0
        y = self.control.height - y
        return (x,y)

    #-------------------------------------------------------------------------
    # Unimplemented or unimplementable methods in Pyglet
    # (These are mostly due to the fact that it is an access layer to GL and
    # not a full GUI toolkit.)
    #-------------------------------------------------------------------------

    def set_tooltip(self, tooltip):
        "Set the current tooltip for the window"
        raise NotImplementedError("No equivalent for set_tooltip() in Pyglet.")

    def create_menu(self, menu_definition, owner):
        "Create a Menu from a string description"
        raise NotImplementedError("create_menu() is not implemented in Pyglet backend.")

    def popup_menu(self, menu, x, y):
        "Pop-up a Menu at a specified location"
        raise NotImplementedError("popup_menu() is not implemented in Pyglet backend.")
