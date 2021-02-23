# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
""" Defines the Label class.
"""

# Major library imports
from math import pi
from numpy import asarray

# Enthought library imports
from kiva.api import FILL, STROKE
from kiva.trait_defs.api import KivaFont
from traits.api import Bool, Enum, Float, HasTraits, Int, List, Str

# Local, relative imports
from .colors import black_color_trait, transparent_color_trait
from .component import Component


class Label(Component):
    """ A text label """

    # The label text.  Carriage returns (\n) are always connverted into
    # line breaks.
    text = Str

    # The angle of rotation of the label.  Only multiples of 90 are supported.
    rotate_angle = Float(0)

    # The color of the label text.
    color = black_color_trait

    # The background color of the label.
    bgcolor = transparent_color_trait

    # The width of the label border. If it is 0, then it is not shown.
    border_width = Int(0)

    # The color of the border.
    border_color = black_color_trait

    # The font of the label text.
    font = KivaFont("modern 10")

    # Number of pixels of margin around the label, for both X and Y dimensions.
    margin = Int(2)

    # Number of pixels of spacing between lines of text.
    line_spacing = Int(5)

    # The horizontal placement of text within the bounds of the label
    hjustify = Enum("left", "center", "right")

    # The vertical placement of text within the bounds of the label
    vjustify = Enum("bottom", "center", "top")

    # By default, labels are not resizable
    resizable = ""

    # ------------------------------------------------------------------------
    # Private traits
    # ------------------------------------------------------------------------

    _bounding_box = List()
    _position_cache_valid = Bool(False)

    def __init__(self, text="", **kwtraits):
        if "text" not in kwtraits:
            kwtraits["text"] = text
        HasTraits.__init__(self, **kwtraits)
        self._bounding_box = [0, 0]

    def _calc_line_positions(self, gc):
        if not self._position_cache_valid:
            with gc:
                gc.set_font(self.font)
                # The bottommost line starts at postion (0,0).
                x_pos = []
                y_pos = []
                self._bounding_box = [0, 0]
                margin = self.margin
                prev_y_pos = margin
                prev_y_height = -self.line_spacing
                max_width = 0
                for line in self.text.split("\n")[::-1]:
                    if line != "":
                        (
                            width,
                            height,
                            descent,
                            leading,
                        ) = gc.get_full_text_extent(line)
                        if width > max_width:
                            max_width = width
                        new_y_pos = (
                            prev_y_pos
                            + prev_y_height
                            - descent
                            + self.line_spacing
                        )
                    else:
                        # For blank lines, we use the height of the previous
                        # line, if there is one.  The width is 0.
                        leading = 0
                        if prev_y_height != -self.line_spacing:
                            new_y_pos = (
                                prev_y_pos + prev_y_height + self.line_spacing
                            )
                            height = prev_y_height
                        else:
                            new_y_pos = prev_y_pos
                            height = 0
                    x_pos.append(-leading + margin)
                    y_pos.append(new_y_pos)
                    prev_y_pos = new_y_pos
                    prev_y_height = height

            width = max_width + 2 * margin + 2 * self.border_width
            height = (
                prev_y_pos + prev_y_height + margin + 2 * self.border_width
            )
            self._bounding_box = [width, height]

            if self.hjustify == "left":
                x_pos = x_pos[::-1]
            else:
                x_pos = asarray(x_pos[::-1], dtype=float)
                if self.hjustify == "center":
                    x_pos += (self.width - width) / 2.0
                elif self.hjustify == "right":
                    x_pos += self.width - width
            self._line_xpos = x_pos

            if self.vjustify == "bottom":
                y_pos = y_pos[::-1]
            else:
                y_pos = asarray(y_pos[::-1], dtype=float)
                if self.vjustify == "center":
                    y_pos += (self.height - height) / 2.0
                elif self.vjustify == "top":
                    y_pos += self.height - height
            self._line_ypos = y_pos

            self._position_cache_valid = True

    def get_width_height(self, gc):
        """ Returns the width and height of the label, in the rotated frame of
        reference.
        """
        self._calc_line_positions(gc)
        width, height = self._bounding_box
        return width, height

    def get_bounding_box(self, gc):
        """ Returns a rectangular bounding box for the Label as (width,height).
        """
        # FIXME: Need to deal with non 90 deg rotations
        width, height = self.get_width_height(gc)
        if self.rotate_angle in (90.0, 270.0):
            return (height, width)
        elif self.rotate_angle in (0.0, 180.0):
            return (width, height)
        else:
            raise NotImplementedError

    def get_bounding_poly(self, gc):
        """
        Returns a list [(x0,y0), (x1,y1),...] of tuples representing a polygon
        that bounds the label.
        """
        raise NotImplementedError

    def _draw_mainlayer(self, gc, view_bounds=None, mode="normal"):
        """ Draws the label.

        This method assumes the graphics context has been translated to the
        correct position such that the origin is at the lower left-hand corner
        of this text label's box.
        """
        # For this version we're not supporting rotated text.
        # temp modified for only one line

        self._calc_line_positions(gc)
        with gc:
            gc.translate_ctm(*self.position)

            # Draw border and fill background
            width, height = self._bounding_box
            if self.bgcolor != "transparent":
                gc.set_fill_color(self.bgcolor_)
                gc.draw_rect((0, 0, width, height), FILL)
            if self.border_width > 0:
                gc.set_stroke_color(self.border_color_)
                gc.set_line_width(self.border_width)
                border_offset = (self.border_width - 1) / 2.0
                gc.draw_rect(
                    (
                        border_offset,
                        border_offset,
                        width - 2 * border_offset,
                        height - 2 * border_offset,
                    ),
                    STROKE,
                )

            gc.set_fill_color(self.color_)
            gc.set_stroke_color(self.color_)
            gc.set_font(self.font)
            if self.font.size <= 8.0:
                gc.set_antialias(0)
            else:
                gc.set_antialias(1)

            gc.rotate_ctm(pi / 180.0 * self.rotate_angle)

            # margin = self.margin
            lines = self.text.split("\n")
            gc.translate_ctm(self.border_width, self.border_width)
            width, height = self.get_width_height(gc)

            for i, line in enumerate(lines):
                if line == "":
                    continue

                if self.rotate_angle == 90.0 or self.rotate_angle == 270.0:
                    x_offset = round(self._line_ypos[i])
                    # this should really be "... - height/2" but
                    # that looks wrong
                    y_offset = round(self._line_xpos[i] - height)
                else:
                    x_offset = round(self._line_xpos[i])
                    y_offset = round(self._line_ypos[i])
                gc.set_text_position(0, 0)
                gc.translate_ctm(x_offset, y_offset)

                gc.show_text(line)
                gc.translate_ctm(-x_offset, -y_offset)

    def _font_changed(self):
        self._position_cache_valid = False

    def _margin_changed(self):
        self._position_cache_valid = False

    def _text_changed(self):
        self._position_cache_valid = False

    def _rotate_angle_changed(self):
        self._position_cache_valid = False
