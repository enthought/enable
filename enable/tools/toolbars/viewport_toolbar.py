# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!

# Library imports

# Enthought Library imports
from enable.api import AbstractOverlay, Container, ColorTrait
from enable.font_metrics_provider import font_metrics_provider
from traits.api import Enum, Bool, Float, Int, Type, List

# Local imports
from .toolbar_buttons import Button


class ViewportToolbar(Container, AbstractOverlay):
    """
    ViewportToolbar is a toolbar that is attached to the top of a viewport on
    the block canvas.
    """

    button_spacing = Int(5)

    button_vposition = Int(6)

    always_on = Bool(True)

    toolbar_height = Float(30.0)

    order = Enum("left-to-right", "right-to-left")

    buttons = List(Type(Button))

    # Override default values for inherited traits
    auto_size = False

    bgcolor = ColorTrait((0.5, 0.5, 0.5, 0.25))

    def __init__(self, component=None, *args, **kw):
        # self.component should be a CanvasViewport
        self.component = component
        for buttontype in self.buttons:
            self.add_button(buttontype())
        super(ViewportToolbar, self).__init__(*args, **kw)

    def _do_layout(self, component=None):
        if component is None:
            component = self.component

        if component is not None:
            self.x = component.x
            # FIXME: Adding 2 to the self.y because there is a tiny gap
            # at the top of the toolbar where components from the block
            # canvas show through.
            self.y = component.y2 - self.toolbar_height + 2
            self.height = self.toolbar_height
            self.width = component.width

        metrics = font_metrics_provider()
        if self.order == "right-to-left":
            last_button_position = self.width - self.button_spacing
            for b in self.components:
                x, y, w, h = metrics.get_text_extent(b.label)
                b.width = w + 2 * b.label_padding
                b.x = last_button_position - b.width
                b.y = self.button_vposition
                last_button_position -= b.width + self.button_spacing * 2
        else:
            last_button_position = 0
            for b in self.components:
                x, y, w, h = metrics.get_text_extent(b.label)
                b.width = w + 2 * b.label_padding
                b.x = self.button_spacing + last_button_position
                b.y = self.button_vposition
                last_button_position += b.width + self.button_spacing * 2

    def overlay(self, other_component, gc, view_bounds=None, mode="normal"):
        c = other_component
        self.do_layout(component=c)
        with gc:
            gc.clip_to_rect(c.x, c.y, c.width, c.height)
            Container._draw(self, gc, view_bounds)

    def add_button(self, button):
        self.add(button)
        button.toolbar_overlay = self
        self._layout_needed = True


class HoverToolbar(ViewportToolbar):

    # This is sort of a hack to make the container handle events like a plain
    # container.  We also want this overlay to be "opaque", so that events
    # inside it don't continue propagating.
    def _dispatch_stateful_event(self, event, suffix):
        Container._dispatch_stateful_event(self, event, suffix)
        if not event.handled:
            if self.is_in(event.x, event.y):
                event.handled = True

    def _container_handle_mouse_event(self, event, suffix):
        if not self.is_in(event.x, event.y) and self.component.auto_hide:
            self.component.remove_toolbar()
            self.component.request_redraw()
