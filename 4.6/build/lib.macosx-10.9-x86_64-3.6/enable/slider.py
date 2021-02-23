# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
from numpy import linspace, zeros

# Enthought library imports
from kiva.api import STROKE
from traits.api import (
    Any, Bool, Enum, Float, Int, Property, Trait, on_trait_change,
)
from traitsui.api import EnumEditor

# Local, relative imports
from .colors import ColorTrait
from .component import Component
from .markers import CustomMarker, MarkerNameDict, marker_names

slider_marker_names = list(marker_names) + ["rect"]
SliderMarkerTrait = Trait(
    "rect",
    "rect",
    MarkerNameDict,
    editor=EnumEditor(values=slider_marker_names),
)


class Slider(Component):
    """ A horizontal or vertical slider bar """

    # ------------------------------------------------------------------------
    # Model traits
    # ------------------------------------------------------------------------

    min = Float()

    max = Float()

    value = Float()

    # The number of ticks to show on the slider.
    num_ticks = Int(4)

    # ------------------------------------------------------------------------
    # Bar and endcap appearance
    # ------------------------------------------------------------------------

    # Whether this is a horizontal or vertical slider
    orientation = Enum("h", "v")

    # The thickness, in pixels, of the lines used to render the ticks,
    # endcaps, and main slider bar.
    bar_width = Int(4)

    bar_color = ColorTrait("black")

    # Whether or not to render endcaps on the slider bar
    endcaps = Bool(True)

    # The extent of the endcaps, in pixels.  This is a read-only property,
    # since the endcap size can be set as either a fixed number of pixels or
    # a percentage of the widget's size in the transverse direction.
    endcap_size = Property

    # The extent of the tickmarks, in pixels.  This is a read-only property,
    # since the endcap size can be set as either a fixed number of pixels or
    # a percentage of the widget's size in the transverse direction.
    tick_size = Property

    # ------------------------------------------------------------------------
    # Slider appearance
    # ------------------------------------------------------------------------

    # The kind of marker to use for the slider.
    slider = SliderMarkerTrait("rect")

    # If the slider marker is "rect", this is the thickness of the slider,
    # i.e. its extent in the dimension parallel to the long axis of the widget.
    # For other slider markers, this has no effect.
    slider_thickness = Int(9)

    # The size of the slider, in pixels.  This is a read-only property, since
    # the slider size can be set as either a fixed number of pixels or a
    # percentage of the widget's size in the transverse direction.
    slider_size = Property

    # For slider markers with a filled area, this is the color of the filled
    # area.  For slider markers that are just lines/strokes (e.g. cross, plus),
    # this is the color of the stroke.
    slider_color = ColorTrait("red")

    # For slider markers with a filled area, this is the color of the outline
    # border drawn around the filled area.  For slider markers that have just
    # lines/strokes, this has no effect.
    slider_border = ColorTrait("none")

    # For slider markers with a filled area, this is the width, in pixels,
    # of the outline around the area.  For slider markers that are just lines/
    # strokes, this is the thickness of the stroke.
    slider_outline_width = Int(1)

    # The kiva.CompiledPath representing the custom path to render for the
    # slider, if the **slider** trait is set to "custom".
    custom_slider = Any()

    # ------------------------------------------------------------------------
    # Interaction traits
    # ------------------------------------------------------------------------

    # Can this slider be interacted with, or is it just a display
    interactive = Bool(True)

    mouse_button = Enum("left", "right")

    event_state = Enum("normal", "dragging")

    # ------------------------------------------------------------------------
    # Private traits
    # ------------------------------------------------------------------------

    # Returns the coordinate index (0 or 1) corresponding to our orientation.
    # Used internally; read-only property.
    axis_ndx = Property()

    _slider_size_mode = Enum("fixed", "percent")
    _slider_percent = Float(0.0)
    _cached_slider_size = Int(10)

    _endcap_size_mode = Enum("fixed", "percent")
    _endcap_percent = Float(0.0)
    _cached_endcap_size = Int(20)

    _tick_size_mode = Enum("fixed", "percent")
    _tick_size_percent = Float(0.0)
    _cached_tick_size = Int(20)

    # A tuple of (dx, dy) of the difference between the mouse position and
    # center of the slider.
    _offset = Any((0, 0))

    def set_range(self, min, max):
        self.min = min
        self.max = max

    def map_screen(self, val):
        """ Returns an (x,y) coordinate corresponding to the location of
        **val** on the slider.
        """
        # Some local variables to handle orientation dependence
        axis_ndx = self.axis_ndx
        other_ndx = 1 - axis_ndx
        screen_low = self.position[axis_ndx]
        screen_high = screen_low + self.bounds[axis_ndx]

        # The return coordinate.  The return value along the non-primary
        # axis will be the same in all cases.
        coord = [0, 0]
        coord[other_ndx] = (
            self.position[other_ndx] + self.bounds[other_ndx] / 2
        )

        # Handle exceptional/boundary cases
        if val <= self.min:
            coord[axis_ndx] = screen_low
            return coord
        elif val >= self.max:
            coord[axis_ndx] = screen_high
            return coord
        elif self.min == self.max:
            coord[axis_ndx] = (screen_low + screen_high) / 2
            return coord

        # Handle normal cases
        coord[axis_ndx] = (val - self.min) / (
            self.max - self.min
        ) * self.bounds[axis_ndx] + screen_low
        return coord

    def map_data(self, x, y, clip=True):
        """ Returns a value between min and max that corresponds to the given
        x and y values.

        Parameters
        ==========
        x, y : Float
            The screen coordinates to map
        clip : Bool (default=True)
            Whether points outside the range should be clipped to the max
            or min value of the slider (depending on which it's closer to)

        Returns
        =======
        value : Float
        """
        # Some local variables to handle orientation dependence
        axis_ndx = self.axis_ndx
        screen_low = self.position[axis_ndx]
        screen_high = screen_low + self.bounds[axis_ndx]
        if self.orientation == "h":
            coord = x
        else:
            coord = y

        # Handle exceptional/boundary cases
        if coord >= screen_high:
            return self.max
        elif coord <= screen_low:
            return self.min
        elif screen_high == screen_low:
            return (self.max + self.min) / 2

        # Handle normal cases
        return (coord - screen_low) / self.bounds[axis_ndx] * (
            self.max - self.min
        ) + self.min

    def set_slider_pixels(self, pixels):
        """ Sets the width of the slider to be a fixed number of pixels

        Parameters
        ==========
        pixels : int
            The number of pixels wide that the slider should be
        """
        self._slider_size_mode = "fixed"
        self._cached_slider_size = pixels

    def set_slider_percent(self, percent):
        """ Sets the width of the slider to be a percentage of the width
        of the slider widget.

        Parameters
        ==========
        percent : float
            The percentage, between 0.0 and 1.0
        """
        self._slider_size_mode = "percent"
        self._slider_percent = percent
        self._update_sizes()

    def set_endcap_pixels(self, pixels):
        """ Sets the width of the endcap to be a fixed number of pixels

        Parameters
        ==========
        pixels : int
            The number of pixels wide that the endcap should be
        """
        self._endcap_size_mode = "fixed"
        self._cached_endcap_size = pixels

    def set_endcap_percent(self, percent):
        """ Sets the width of the endcap to be a percentage of the width
        of the endcap widget.

        Parameters
        ==========
        percent : float
            The percentage, between 0.0 and 1.0
        """
        self._endcap_size_mode = "percent"
        self._endcap_percent = percent
        self._update_sizes()

    def set_tick_pixels(self, pixels):
        """ Sets the width of the tick marks to be a fixed number of pixels

        Parameters
        ==========
        pixels : int
            The number of pixels wide that the endcap should be
        """
        self._tick_size_mode = "fixed"
        self._cached_tick_size = pixels

    def set_tick_percent(self, percent):
        """ Sets the width of the tick marks to be a percentage of the width
        of the endcap widget.

        Parameters
        ==========
        percent : float
            The percentage, between 0.0 and 1.0
        """
        self._tick_size_mode = "percent"
        self._tick_percent = percent
        self._update_sizes()

    # ------------------------------------------------------------------------
    # Rendering methods
    # ------------------------------------------------------------------------

    def _draw_mainlayer(self, gc, view_bounds=None, mode="normal"):
        bar_x = self.x + self.width / 2
        bar_y = self.y + self.height / 2

        # Draw the bar and endcaps
        gc.set_stroke_color(self.bar_color_)
        gc.set_line_width(self.bar_width)
        if self.orientation == "h":
            gc.move_to(self.x, bar_y)
            gc.line_to(self.x2, bar_y)
            gc.stroke_path()
            if self.endcaps:
                start_y = bar_y - self._cached_endcap_size / 2
                end_y = bar_y + self._cached_endcap_size / 2
                gc.move_to(self.x, start_y)
                gc.line_to(self.x, end_y)
                gc.move_to(self.x2, start_y)
                gc.line_to(self.x2, end_y)
            if self.num_ticks > 0:
                x_pts = linspace(
                    self.x, self.x2, self.num_ticks + 2
                ).astype(int)
                starts = zeros((len(x_pts), 2), dtype=int)
                starts[:, 0] = x_pts
                starts[:, 1] = bar_y - self._cached_tick_size / 2
                ends = starts.copy()
                ends[:, 1] = bar_y + self._cached_tick_size / 2
                gc.line_set(starts, ends)
        else:
            gc.move_to(bar_x, self.y)
            gc.line_to(bar_x, self.y2)
            if self.endcaps:
                start_x = bar_x - self._cached_endcap_size / 2
                end_x = bar_x + self._cached_endcap_size / 2
                gc.move_to(start_x, self.y)
                gc.line_to(end_x, self.y)
                gc.move_to(start_x, self.y2)
                gc.line_to(end_x, self.y2)
            if self.num_ticks > 0:
                y_pts = linspace(
                    self.y, self.y2, self.num_ticks + 2
                ).astype(int)
                starts = zeros((len(y_pts), 2), dtype=int)
                starts[:, 1] = y_pts
                starts[:, 0] = bar_x - self._cached_tick_size / 2
                ends = starts.copy()
                ends[:, 0] = bar_x + self._cached_tick_size / 2
                gc.line_set(starts, ends)
        gc.stroke_path()

        # Draw the slider
        pt = self.map_screen(self.value)
        if self.slider == "rect":
            gc.set_fill_color(self.slider_color_)
            gc.set_stroke_color(self.slider_border_)
            gc.set_line_width(self.slider_outline_width)
            rect = self._get_rect_slider_bounds()
            gc.rect(*rect)
            gc.draw_path()
        else:
            self._render_marker(
                gc,
                pt,
                self._cached_slider_size,
                self.slider_(),
                self.custom_slider,
            )

    def _get_rect_slider_bounds(self):
        """ Returns the (x, y, w, h) bounds of the rectangle representing the
        slider.

        Used for rendering and hit detection.
        """
        bar_x = self.x + self.width / 2
        bar_y = self.y + self.height / 2
        pt = self.map_screen(self.value)
        if self.orientation == "h":
            slider_height = self._cached_slider_size
            return (
                pt[0] - self.slider_thickness,
                bar_y - slider_height / 2,
                self.slider_thickness,
                slider_height,
            )
        else:
            slider_width = self._cached_slider_size
            return (
                bar_x - slider_width / 2,
                pt[1] - self.slider_thickness,
                slider_width,
                self.slider_thickness,
            )

    def _render_marker(self, gc, point, size, marker, custom_path):
        with gc:
            gc.begin_path()
            if marker.draw_mode == STROKE:
                gc.set_stroke_color(self.slider_color_)
                gc.set_line_width(self.slider_thickness)
            else:
                gc.set_fill_color(self.slider_color_)
                gc.set_stroke_color(self.slider_border_)
                gc.set_line_width(self.slider_outline_width)

            if (hasattr(gc, "draw_marker_at_points")
                    and (marker.__class__ != CustomMarker)
                    and gc.draw_marker_at_points(
                        [point], size, marker.kiva_marker
                    ) != 0):
                pass
            elif hasattr(gc, "draw_path_at_points"):
                if marker.__class__ != CustomMarker:
                    path = gc.get_empty_path()
                    marker.add_to_path(path, size)
                    mode = marker.draw_mode
                else:
                    path = custom_path
                    mode = STROKE
                if not marker.antialias:
                    gc.set_antialias(False)
                gc.draw_path_at_points([point], path, mode)
            else:
                if not marker.antialias:
                    gc.set_antialias(False)
                if marker.__class__ != CustomMarker:
                    gc.translate_ctm(*point)
                    # Kiva GCs have a path-drawing interface
                    marker.add_to_path(gc, size)
                    gc.draw_path(marker.draw_mode)
                else:
                    path = custom_path
                    gc.translate_ctm(*point)
                    gc.add_path(path)
                    gc.draw_path(STROKE)

    # ------------------------------------------------------------------------
    # Interaction event handlers
    # ------------------------------------------------------------------------

    def normal_left_down(self, event):
        if self.mouse_button == "left":
            return self._mouse_pressed(event)

    def dragging_left_up(self, event):
        if self.mouse_button == "left":
            return self._mouse_released(event)

    def normal_right_down(self, event):
        if self.mouse_button == "right":
            return self._mouse_pressed(event)

    def dragging_right_up(self, event):
        if self.mouse_button == "right":
            return self._mouse_released(event)

    def dragging_mouse_move(self, event):
        dx, dy = self._offset
        self.value = self.map_data(event.x - dx, event.y - dy)
        event.handled = True
        self.request_redraw()

    def dragging_mouse_leave(self, event):
        self.event_state = "normal"

    def _mouse_pressed(self, event):
        # Determine the slider bounds so we can hit test it
        pt = self.map_screen(self.value)
        if self.slider == "rect":
            x, y, w, h = self._get_rect_slider_bounds()
            x2 = x + w
            y2 = y + h
        else:
            x, y = pt
            size = self._cached_slider_size
            x -= size / 2
            y -= size / 2
            x2 = x + size
            y2 = y + size

        # Hit test both the slider and against the bar.  If the user has
        # clicked on the bar but outside of the slider, we set the _offset
        # and call dragging_mouse_move() to teleport the slider to the
        # mouse click position.
        if self.orientation == "v" and (x <= event.x <= x2):
            if not (y <= event.y <= y2):
                self._offset = (event.x - pt[0], 0)
                self.dragging_mouse_move(event)
            else:
                self._offset = (event.x - pt[0], event.y - pt[1])
        elif self.orientation == "h" and (y <= event.y <= y2):
            if not (x <= event.x <= x2):
                self._offset = (0, event.y - pt[1])
                self.dragging_mouse_move(event)
            else:
                self._offset = (event.x - pt[0], event.y - pt[1])
        else:
            # The mouse click missed the bar and the slider.
            return

        event.handled = True
        self.event_state = "dragging"

    def _mouse_released(self, event):
        self.event_state = "normal"
        event.handled = True

    # ------------------------------------------------------------------------
    # Private trait event handlers and property getters/setters
    # ------------------------------------------------------------------------

    def _get_axis_ndx(self):
        if self.orientation == "h":
            return 0
        else:
            return 1

    def _get_slider_size(self):
        return self._cached_slider_size

    def _get_endcap_size(self):
        return self._cached_endcap_size

    def _get_tick_size(self):
        return self._cached_tick_size

    @on_trait_change("bounds,bounds_items")
    def _update_sizes(self):
        if self._slider_size_mode == "percent":
            if self.orientation == "h":
                self._cached_slider_size = int(
                    self.height * self._slider_percent
                )
            else:
                self._cached_slider_size = int(
                    self.width * self._slider_percent
                )
        if self._endcap_size_mode == "percent":
            if self.orientation == "h":
                self._cached_endcap_size = int(
                    self.height * self._endcap_percent
                )
            else:
                self._cached_endcap_size = int(
                    self.width * self._endcap_percent
                )
