# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
"""
TextGrid is a text grid widget that is meant to be used with Numpy.
"""

# Major library imports
from numpy import arange, array, dstack, repeat, newaxis

# Enthought library imports
from traits.api import (
    Any, Array, Bool, Int, List, Property, Trait, Tuple, on_trait_change,
)
from kiva.trait_defs.api import KivaFont

# Relative imports
from .component import Component
from .colors import black_color_trait, ColorTrait
from .enable_traits import LineStyle
from .font_metrics_provider import font_metrics_provider


class TextGrid(Component):
    """
    A 2D grid of string values
    """

    # A 2D array of strings
    string_array = Array

    # The cell size can be set to a tuple (w,h) or to "auto".
    cell_size = Property

    # ------------------------------------------------------------------------
    # Appereance traits
    # ------------------------------------------------------------------------

    # The font to use for the text of the grid
    font = KivaFont("modern 14")

    # The color of the text
    text_color = black_color_trait

    # The padding around each cell
    cell_padding = Int(5)

    # The thickness of the border between cells
    cell_border_width = Int(1)

    # The color of the border between cells
    cell_border_color = black_color_trait

    # The dash style of the border between cells
    cell_border_style = LineStyle("solid")

    # Text color of highlighted items
    highlight_color = ColorTrait("red")

    # Cell background color of highlighted items
    highlight_bgcolor = ColorTrait("lightgray")

    # A list of tuples of the (i,j) of selected cells
    selected_cells = List

    # ------------------------------------------------------------------------
    # Private traits
    # ------------------------------------------------------------------------

    # Are our cached extent values still valid?
    _cache_valid = Bool(False)

    # The maximum width and height of all cells, as a tuple (w,h)
    _cached_cell_size = Tuple

    # The maximum (leading, descent) of all the text strings (positive value)
    _text_offset = Array

    # An array NxMx2 of the x,y positions of the lower-left coordinates of
    # each cell
    _cached_cell_coords = Array

    # "auto" or a tuple
    _cell_size = Trait("auto", Any)

    # ------------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------------

    def __init__(self, **kwtraits):
        super(Component, self).__init__(**kwtraits)
        self.selected_cells = []

    # ------------------------------------------------------------------------
    # AbstractComponent interface
    # ------------------------------------------------------------------------

    def _draw_mainlayer(self, gc, view_bounds=None, mode="default"):
        text_color = self.text_color_
        highlight_color = self.highlight_color_
        highlight_bgcolor = self.highlight_bgcolor_
        padding = self.cell_padding
        border_width = self.cell_border_width

        with gc:
            gc.set_stroke_color(text_color)
            gc.set_fill_color(text_color)
            gc.set_font(self.font)
            gc.set_text_position(0, 0)

            width, height = self._get_actual_cell_size()
            numrows, numcols = self.string_array.shape

            # draw selected backgrounds
            # XXX should this be in the background layer?
            for j, row in enumerate(self.string_array):
                for i, text in enumerate(row):
                    if (i, j) in self.selected_cells:
                        gc.set_fill_color(highlight_bgcolor)
                        ll_x, ll_y = self._cached_cell_coords[i, j + 1]
                        # render this a bit big, but covered by border
                        gc.rect(
                            ll_x,
                            ll_y,
                            width + 2 * padding + border_width,
                            height + 2 * padding + border_width,
                        )
                        gc.fill_path()
                        gc.set_fill_color(text_color)

            self._draw_grid_lines(gc)

            for j, row in enumerate(self.string_array):
                for i, text in enumerate(row):
                    x, y = (
                        self._cached_cell_coords[i, j + 1]
                        + self._text_offset
                        + padding
                        + border_width / 2.0
                    )

                    if (i, j) in self.selected_cells:
                        gc.set_fill_color(highlight_color)
                        gc.set_stroke_color(highlight_color)
                        gc.set_text_position(x, y)
                        gc.show_text(text)
                        gc.set_stroke_color(text_color)
                        gc.set_fill_color(text_color)
                    else:
                        gc.set_text_position(x, y)
                        gc.show_text(text)

    # ------------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------------

    def _draw_grid_lines(self, gc):
        gc.set_stroke_color(self.cell_border_color_)
        gc.set_line_dash(self.cell_border_style_)
        gc.set_line_width(self.cell_border_width)

        # Skip the leftmost and bottommost cell coords (since Y axis is
        # reversed, the bottommost coord is the last one)
        x_points = self._cached_cell_coords[:, 0, 0]
        y_points = self._cached_cell_coords[0, :, 1]

        for x in x_points:
            gc.move_to(x, self.y)
            gc.line_to(x, self.y + self.height)
            gc.stroke_path()

        for y in y_points:
            gc.move_to(self.x, y)
            gc.line_to(self.x + self.width, y)
            gc.stroke_path()

    def _compute_cell_sizes(self):
        if not self._cache_valid:
            gc = font_metrics_provider()
            max_w = 0
            max_h = 0
            min_l = 0
            min_d = 0
            for text in self.string_array.ravel():
                gc.set_font(self.font)
                l, d, w, h = gc.get_text_extent(text)
                if -l + w > max_w:
                    max_w = -l + w
                if -d + h > max_h:
                    max_h = -d + h
                if l < min_l:
                    min_l = l
                if d < min_d:
                    min_d = d

            self._cached_cell_size = (max_w, max_h)
            self._text_offset = array([-min_l, -min_d])
            self._cache_valid = True

    def _compute_positions(self):

        if self.string_array is None or len(self.string_array.shape) != 2:
            return

        width, height = self._get_actual_cell_size()
        numrows, numcols = self.string_array.shape

        cell_width = width + 2 * self.cell_padding + self.cell_border_width
        cell_height = height + 2 * self.cell_padding + self.cell_border_width

        x_points = (
            arange(numcols + 1) * cell_width
            + self.cell_border_width / 2.0
            + self.x
        )
        y_points = (
            arange(numrows + 1) * cell_height
            + self.cell_border_width / 2.0
            + self.y
        )

        tmp = dstack(
            (
                repeat(x_points[:, newaxis], numrows + 1, axis=1),
                repeat(y_points[:, newaxis].T, numcols + 1, axis=0),
            )
        )

        # We have to reverse the y-axis (e.g. the 0th row needs to be at the
        # highest y-position).
        self._cached_cell_coords = tmp[:, ::-1]

    def _update_bounds(self):
        if self.string_array is not None and len(self.string_array.shape) == 2:
            rows, cols = self.string_array.shape
            margin = 2 * self.cell_padding + self.cell_border_width
            width, height = self._get_actual_cell_size()
            self.bounds = [
                cols * (width + margin) + self.cell_border_width,
                rows * (height + margin) + self.cell_border_width,
            ]

        else:
            self.bounds = [0, 0]

    def _get_actual_cell_size(self):
        if self._cell_size == "auto":
            if not self._cache_valid:
                self._compute_cell_sizes()
            return self._cached_cell_size

        else:
            if not self._cache_valid:
                # actually computing the text offset
                self._compute_cell_sizes()
            return self._cell_size

    # ------------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------------

    def normal_left_down(self, event):
        self.selected_cells = [self._get_index_for_xy(event.x, event.y)]
        self.request_redraw()

    def _get_index_for_xy(self, x, y):
        width, height = (
            array(self._get_actual_cell_size())
            + 2 * self.cell_padding
            + self.cell_border_width
        )

        numrows, numcols = self.string_array.shape
        i = int((x - self.padding_left) / width)
        j = numrows - (int((y - self.padding_bottom) / height) + 1)
        shape = self.string_array.shape
        if 0 <= i < shape[1] and 0 <= j < shape[0]:
            return i, j
        else:
            return None

    # ------------------------------------------------------------------------
    # Trait events, property setters and getters
    # ------------------------------------------------------------------------

    def _string_array_changed(self, old, new):
        if self._cell_size == "auto":
            self._cache_valid = False
            self._compute_cell_sizes()
        self._compute_positions()
        self._update_bounds()

    @on_trait_change("cell_border_width,cell_padding")
    def cell_properties_changed(self):
        self._compute_positions()
        self._update_bounds()

    def _set_cell_size(self, newsize):
        self._cell_size = newsize
        if newsize == "auto":
            self._compute_cell_sizes()
        self._compute_positions()
        self._update_bounds()

    def _get_cell_size(self):
        return self._cell_size
