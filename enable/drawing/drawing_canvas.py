# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
from enable.api import Container, Component, ColorTrait
from kiva.api import FILL, FILL_STROKE
from kiva.trait_defs.api import KivaFont
from traits.api import Any, Bool, Delegate, Enum, Instance, Int, List, Str


class Button(Component):

    color = ColorTrait("lightblue")

    down_color = ColorTrait("darkblue")

    border_color = ColorTrait("blue")

    label = Str

    label_font = KivaFont("modern 12")

    label_color = ColorTrait("white")

    down_label_color = ColorTrait("white")

    button_state = Enum("up", "down")

    # A reference to the radio group that this button belongs to
    radio_group = Any

    # Default size of the button if no label is present
    bounds = [32, 32]

    # Generally, buttons are not resizable
    resizable = ""

    _got_mousedown = Bool(False)

    def perform(self, event):
        """
        Called when the button is depressed.  'event' is the Enable mouse event
        that triggered this call.
        """
        pass

    def _draw_mainlayer(self, gc, view_bounds, mode="default"):
        if self.button_state == "up":
            self.draw_up(gc, view_bounds)
        else:
            self.draw_down(gc, view_bounds)

    def draw_up(self, gc, view_bounds):
        with gc:
            gc.set_fill_color(self.color_)
            gc.set_stroke_color(self.border_color_)
            gc.draw_rect(
                (
                    int(self.x),
                    int(self.y),
                    int(self.width) - 1,
                    int(self.height) - 1,
                ),
                FILL_STROKE,
            )
            self._draw_label(gc)

    def draw_down(self, gc, view_bounds):
        with gc:
            gc.set_fill_color(self.down_color_)
            gc.set_stroke_color(self.border_color_)
            gc.draw_rect(
                (
                    int(self.x),
                    int(self.y),
                    int(self.width) - 1,
                    int(self.height) - 1,
                ),
                FILL_STROKE,
            )
            self._draw_label(gc, color=self.down_label_color_)

    def _draw_label(self, gc, color=None):
        if self.label != "":
            gc.set_font(self.label_font)
            x, y, w, h = gc.get_text_extent(self.label)
            if color is None:
                color = self.label_color_
            gc.set_fill_color(color)
            gc.set_stroke_color(color)
            gc.show_text(
                self.label,
                (
                    self.x + (self.width - w - x) / 2,
                    self.y + (self.height - h - y) / 2,
                ),
            )

    def normal_left_down(self, event):
        self.button_state = "down"
        self._got_mousedown = True
        self.request_redraw()
        event.handled = True

    def normal_left_up(self, event):
        self.button_state = "up"
        self._got_mousedown = False
        self.request_redraw()
        self.perform(event)
        event.handled = True


class ToolbarButton(Button):

    toolbar = Any

    canvas = Delegate("toolbar")

    def __init__(self, *args, **kw):
        toolbar = kw.pop("toolbar", None)
        super(ToolbarButton, self).__init__(*args, **kw)
        if toolbar:
            self.toolbar = toolbar
            toolbar.add(self)


class DrawingCanvasToolbar(Container):
    """ The tool bar hosts Buttons and also consumes other mouse events, so
    that tools on the underlying canvas don't get them.

    FIXME: Right now this toolbar only supports the addition of buttons, and
           not button removal.  (Why would you ever want to remove a useful
           button?)
    """

    canvas = Instance("DrawingCanvas")

    button_spacing = Int(5)

    auto_size = False

    _last_button_position = Int(0)

    def add_button(self, *buttons):
        for button in buttons:
            self.add(button)
            button.toolbar = self
            # Compute the new position for the button
            button.x = self.button_spacing + self._last_button_position
            self._last_button_position += (
                button.width + self.button_spacing * 2
            )
            button.y = int((self.height - button.height) / 2)

    def _canvas_changed(self, old, new):
        if old:
            old.on_trait_change(
                self._canvas_bounds_changed, "bounds", remove=True
            )
            old.on_trait_change(
                self._canvas_bounds_changed, "bounds_items", remove=True
            )

        if new:
            new.on_trait_change(self._canvas_bounds_changed, "bounds")
            new.on_trait_change(self._canvas_bounds_changed, "bounds_items")

    def _canvas_bounds_changed(self):
        self.width = self.canvas.width
        self.y = self.canvas.height - self.height

    def _dispatch_stateful_event(self, event, suffix):
        super(DrawingCanvasToolbar, self)._dispatch_stateful_event(
            event, suffix
        )
        event.handled = True


class DrawingCanvas(Container):
    """
    A DrawingCanvas has some buttons which toggle what kind of drawing tools
    are active on the canvas, then allow arbitrary painting on the canvas.
    """

    # The active tool is the primary interactor on the canvas.  It gets
    # a chance to handle events before they are passed on to other components
    # and listener tools.
    active_tool = Any

    # Listening tools are always enabled and get all events (unless the active
    # tool has vetoed it), but they cannot prevent other tools from getting
    # events.
    listening_tools = List

    # The background color of the canvas
    bgcolor = ColorTrait("white")

    toolbar = Instance(DrawingCanvasToolbar, args=())

    fit_window = True

    def dispatch(self, event, suffix):
        # See if the event happened on the toolbar:

        event.offset_xy(*self.position)
        if self.toolbar.is_in(event.x, event.y):
            self.toolbar.dispatch(event, suffix)
        event.pop()
        if event.handled:
            return

        if self.active_tool is not None:
            self.active_tool.dispatch(event, suffix)

        if event.handled:
            return

        for tool in self.listening_tools:
            tool.dispatch(event, suffix)

        super(DrawingCanvas, self).dispatch(event, suffix)

    def activate(self, tool):
        """
        Makes the indicated tool the active tool on the canvas and moves the
        current active tool back into the list of tools.
        """
        self.active_tool = tool

    def _draw_container_mainlayer(self, gc, view_bounds=None, mode="default"):
        active_tool = self.active_tool
        if active_tool and active_tool.draw_mode == "exclusive":
            active_tool.draw(gc, view_bounds, mode)
        else:
            # super(DrawingCanvas, self)._draw(gc, view_bounds, mode)
            for tool in self.listening_tools:
                tool.draw(gc, view_bounds, mode)
            if active_tool:
                active_tool.draw(gc, view_bounds, mode)

            self.toolbar.draw(gc, view_bounds, mode)

    def _draw_container_background(self, gc, view_bounds=None, mode="default"):
        if self.bgcolor not in ("clear", "transparent", "none"):
            with gc:
                gc.set_antialias(False)
                gc.set_fill_color(self.bgcolor_)
                gc.draw_rect(
                    (
                        int(self.x),
                        int(self.y),
                        int(self.width) - 1,
                        int(self.height) - 1,
                    ),
                    FILL,
                )

    # ------------------------------------------------------------------------
    # Event listeners
    # ------------------------------------------------------------------------

    def _tools_items_changed(self):
        self.request_redraw()
