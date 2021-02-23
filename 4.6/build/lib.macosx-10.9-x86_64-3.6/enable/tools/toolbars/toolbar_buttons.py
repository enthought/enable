# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!

# Enthought library imports
from enable.api import ColorTrait, Component
from enable.font_metrics_provider import font_metrics_provider
from kiva.trait_defs.api import KivaFont
from traits.api import Bool, Enum, Int, Str, Tuple


class Button(Component):

    color = ColorTrait((0.6, 0.6, 0.6, 1.0))

    down_color = ColorTrait("gray")

    border_color = ColorTrait((0.4, 0.4, 0.4, 1.0))

    # important for rendering rounded buttons properly, since the default for
    # the Component parent class is 'white'
    bgcolor = "clear"

    label = Str

    label_font = KivaFont("modern 11 bold")

    label_color = ColorTrait("white")

    label_shadow = ColorTrait("gray")

    shadow_text = Bool(True)

    label_padding = Int(5)

    height = Int(20)

    button_state = Enum("up", "down")

    end_radius = Int(10)

    # Default size of the button if no label is present
    bounds = [32, 32]

    # Cached value of the measured sizes of self.label
    _text_extents = Tuple

    def perform(self, event):
        """ Called when the button is depressed.

        'event' is the Enable mouse event that triggered this call.
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
            self._draw_actual_button(gc)

    def draw_down(self, gc, view_bounds):
        with gc:
            gc.set_fill_color(self.down_color_)
            self._draw_actual_button(gc)

    def _draw_actual_button(self, gc):
        gc.set_stroke_color(self.border_color_)
        gc.begin_path()

        gc.move_to(self.x + self.end_radius, self.y)

        gc.arc_to(
            self.x + self.width,
            self.y,
            self.x + self.width,
            self.y + self.end_radius,
            self.end_radius,
        )
        gc.arc_to(
            self.x + self.width,
            self.y + self.height,
            self.x + self.width - self.end_radius,
            self.y + self.height,
            self.end_radius,
        )
        gc.arc_to(
            self.x, self.y + self.height, self.x, self.y, self.end_radius
        )
        gc.arc_to(
            self.x,
            self.y,
            self.x + self.width + self.end_radius,
            self.y,
            self.end_radius,
        )

        gc.draw_path()
        self._draw_label(gc)

    def _draw_label(self, gc):
        if self.label != "":
            if self._text_extents is None or len(self._text_extents) == 0:
                self._recompute_font_metrics()
            x, y, w, h = self._text_extents
            gc.set_font(self.label_font)
            text_offset = 0.0

            if self.shadow_text:
                # Draw shadow text
                gc.set_fill_color(self.label_shadow_)
                x_pos = self.x + (self.width - w - x) / 2 + 0.5
                y_pos = self.y + (self.height - h - y) / 2 - 0.5
                gc.show_text_at_point(self.label, x_pos, y_pos)
                text_offset = 0.5

            # Draw foreground text to button
            gc.set_fill_color(self.label_color_)
            x_pos = self.x + (self.width - w - x) / 2 - text_offset
            y_pos = self.y + (self.height - h - y) / 2 + text_offset
            gc.show_text_at_point(self.label, x_pos, y_pos)

    def normal_left_down(self, event):
        self.button_state = "down"
        self.request_redraw()
        event.handled = True

    def normal_left_up(self, event):
        self.button_state = "up"
        self.request_redraw()
        self.perform(event)
        event.handled = True

    def _recompute_font_metrics(self):
        if self.label != "":
            metrics = font_metrics_provider()
            metrics.set_font(self.label_font)
            self._text_extents = metrics.get_text_extent(self.label)

    def _label_font_changed(self, old, new):
        self._recompute_font_metrics()

    def _label_changed(self, old, new):
        self._recompute_font_metrics()


class SampleButtonButton(Button):

    label = Str("Sample Button")

    def perform(self, event):
        print("this button is a sample")
