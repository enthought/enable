# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
import math

from traits.api import Float, Property, List, Str, Range
from enable.api import Component
from kiva.trait_defs.api import KivaFont
from kiva import affine


def percent_to_db(percent):
    if percent == 0.0:
        db = float("-inf")
    else:
        db = 20 * math.log10(percent / 100.0)
    return db


def db_to_percent(db):
    percent = math.pow(10, db / 20.0 + 2)
    return percent


class VUMeter(Component):

    # Value expressed in dB
    db = Property(Float)

    # Value expressed as a percent.
    percent = Range(low=0.0)

    # The maximum value to be display in the VU Meter, expressed as a percent.
    max_percent = Float(150.0)

    # Angle (in degrees) from a horizontal line through the hinge of the
    # needle to the edge of the meter axis.
    angle = Float(45.0)

    # Values of the percentage-based ticks; these are drawn and labeled along
    # the bottom of the curve axis.
    percent_ticks = List(list(range(0, 101, 20)))

    # Text to write in the middle of the VU Meter.
    text = Str("VU")

    # Font used to draw `text`.
    text_font = KivaFont("modern 48")

    # Font for the db tick labels.
    db_tick_font = KivaFont("modern 16")

    # Font for the percent tick labels.
    percent_tick_font = KivaFont("modern 12")

    # beta is the fraction of the of needle that is "hidden".
    # beta == 0 puts the hinge point of the needle on the bottom
    # edge of the window.  Values that result in a decent looking
    # meter are 0 < beta < .65.
    # XXX needs a better name!
    _beta = Float(0.3)

    # _outer_radial_margin is the radial extent beyond the circular axis
    # to include  in calculations of the space required for the meter.
    # This allows room for the ticks and labels.
    _outer_radial_margin = Float(60.0)

    # The angle (in radians) of the span of the curve axis.
    _phi = Property(Float, depends_on=["angle"])

    # This is the radius of the circular axis (in screen coordinates).
    _axis_radius = Property(Float, depends_on=["_phi", "width", "height"])

    # ---------------------------------------------------------------------
    # Trait Property methods
    # ---------------------------------------------------------------------

    def _get_db(self):
        db = percent_to_db(self.percent)
        return db

    def _set_db(self, value):
        self.percent = db_to_percent(value)

    def _get__phi(self):
        phi = math.pi * (180.0 - 2 * self.angle) / 180.0
        return phi

    def _get__axis_radius(self):
        M = self._outer_radial_margin
        beta = self._beta
        w = self.width
        h = self.height
        phi = self._phi

        R1 = w / (2 * math.sin(phi / 2)) - M
        R2 = (h - M) / (1 - beta * math.cos(phi / 2))
        R = min(R1, R2)
        return R

    # ---------------------------------------------------------------------
    # Trait change handlers
    # ---------------------------------------------------------------------

    def _anytrait_changed(self):
        self.request_redraw()

    # ---------------------------------------------------------------------
    # Component API
    # ---------------------------------------------------------------------

    def _draw_mainlayer(self, gc, view_bounds=None, mode="default"):

        beta = self._beta
        phi = self._phi

        w = self.width

        M = self._outer_radial_margin
        R = self._axis_radius

        # (ox, oy) is the position of the "hinge point" of the needle
        # (i.e. the center of rotation).  For beta > ~0, oy is negative,
        # so this point is below the visible region.
        ox = self.x + self.width // 2
        oy = -beta * R * math.cos(phi / 2) + 1

        left_theta = math.radians(180 - self.angle)
        right_theta = math.radians(self.angle)

        # The angle of the 100% position.
        nominal_theta = self._percent_to_theta(100.0)

        # The color of the axis for percent > 100.
        red = (0.8, 0, 0)

        with gc:
            gc.set_antialias(True)

            # Draw everything relative to the center of the circles.
            gc.translate_ctm(ox, oy)

            # Draw the primary ticks and tick labels on the curved axis.
            gc.set_fill_color((0, 0, 0))
            gc.set_font(self.db_tick_font)
            for db in [-20, -10, -7, -5, -3, -2, -1, 0, 1, 2, 3]:
                db_percent = db_to_percent(db)
                theta = self._percent_to_theta(db_percent)
                x1 = R * math.cos(theta)
                y1 = R * math.sin(theta)
                x2 = (R + 0.3 * M) * math.cos(theta)
                y2 = (R + 0.3 * M) * math.sin(theta)
                gc.set_line_width(2.5)
                gc.move_to(x1, y1)
                gc.line_to(x2, y2)
                gc.stroke_path()

                text = str(db)
                if db > 0:
                    text = "+" + text
                self._draw_rotated_label(gc, text, theta, R + 0.4 * M)

            # Draw the secondary ticks on the curve axis.
            for db in [-15, -9, -8, -6, -4, -0.5, 0.5]:
                # db_percent = 100 * math.pow(10.0, db / 20.0)
                db_percent = db_to_percent(db)
                theta = self._percent_to_theta(db_percent)
                x1 = R * math.cos(theta)
                y1 = R * math.sin(theta)
                x2 = (R + 0.2 * M) * math.cos(theta)
                y2 = (R + 0.2 * M) * math.sin(theta)
                gc.set_line_width(1.0)
                gc.move_to(x1, y1)
                gc.line_to(x2, y2)
                gc.stroke_path()

            # Draw the percent ticks and label on the bottom of the
            # curved axis.
            gc.set_font(self.percent_tick_font)
            gc.set_fill_color((0.5, 0.5, 0.5))
            gc.set_stroke_color((0.5, 0.5, 0.5))
            percents = self.percent_ticks
            for tick_percent in percents:
                theta = self._percent_to_theta(tick_percent)
                x1 = (R - 0.15 * M) * math.cos(theta)
                y1 = (R - 0.15 * M) * math.sin(theta)
                x2 = R * math.cos(theta)
                y2 = R * math.sin(theta)
                gc.set_line_width(2.0)
                gc.move_to(x1, y1)
                gc.line_to(x2, y2)
                gc.stroke_path()

                text = str(tick_percent)
                if tick_percent == percents[-1]:
                    text = text + "%"
                self._draw_rotated_label(gc, text, theta, R - 0.3 * M)

            if self.text:
                gc.set_font(self.text_font)
                tx, ty, tw, th = gc.get_text_extent(self.text)
                gc.set_fill_color((0, 0, 0, 0.25))
                gc.set_text_matrix(affine.affine_from_rotation(0))
                gc.set_text_position(-0.5 * tw, (0.75 * beta + 0.25) * R)
                gc.show_text(self.text)

            # Draw the red curved axis.
            gc.set_stroke_color(red)
            w = 10
            gc.set_line_width(w)
            gc.arc(0, 0, R + 0.5 * w - 1, right_theta, nominal_theta)
            gc.stroke_path()

            # Draw the black curved axis.
            w = 4
            gc.set_line_width(w)
            gc.set_stroke_color((0, 0, 0))
            gc.arc(0, 0, R + 0.5 * w - 1, nominal_theta, left_theta)
            gc.stroke_path()

            # Draw the filled arc at the bottom.
            gc.set_line_width(2)
            gc.set_stroke_color((0, 0, 0))
            gc.arc(
                0,
                0,
                beta * R,
                math.radians(self.angle),
                math.radians(180 - self.angle),
            )
            gc.stroke_path()
            gc.set_fill_color((0, 0, 0, 0.25))
            gc.arc(
                0,
                0,
                beta * R,
                math.radians(self.angle),
                math.radians(180 - self.angle),
            )
            gc.fill_path()

            # Draw the needle.
            percent = self.percent
            # If percent exceeds max_percent, the needle is drawn at
            # max_percent.
            if percent > self.max_percent:
                percent = self.max_percent
            needle_theta = self._percent_to_theta(percent)
            gc.rotate_ctm(needle_theta - 0.5 * math.pi)
            self._draw_vertical_needle(gc)

    # ---------------------------------------------------------------------
    # Private methods
    # ---------------------------------------------------------------------

    def _draw_vertical_needle(self, gc):
        """ Draw the needle of the meter, pointing straight up. """
        beta = self._beta
        R = self._axis_radius
        end_y = beta * R
        blob_y = R - 0.6 * self._outer_radial_margin
        tip_y = R + 0.2 * self._outer_radial_margin
        lw = 5

        with gc:
            gc.set_alpha(1)
            gc.set_fill_color((0, 0, 0))

            # Draw the needle from the bottom to the blob.
            gc.set_line_width(lw)
            gc.move_to(0, end_y)
            gc.line_to(0, blob_y)
            gc.stroke_path()

            # Draw the thin part of the needle from the blob to the tip.
            gc.move_to(lw, blob_y)
            control_y = blob_y + 0.25 * (tip_y - blob_y)
            gc.quad_curve_to(0.2 * lw, control_y, 0, tip_y)
            gc.quad_curve_to(-0.2 * lw, control_y, -lw, blob_y)
            gc.line_to(lw, blob_y)
            gc.fill_path()

            # Draw the blob on the needle.
            gc.arc(0, blob_y, 6.0, 0, 2 * math.pi)
            gc.fill_path()

    def _draw_rotated_label(self, gc, text, theta, radius):

        tx, ty, tw, th = gc.get_text_extent(text)

        rr = math.sqrt(radius ** 2 + (0.5 * tw) ** 2)
        dtheta = math.atan2(0.5 * tw, radius)
        text_theta = theta + dtheta
        x = rr * math.cos(text_theta)
        y = rr * math.sin(text_theta)

        rot_theta = theta - 0.5 * math.pi
        with gc:
            gc.set_text_matrix(affine.affine_from_rotation(rot_theta))
            gc.set_text_position(x, y)
            gc.show_text(text)

    def _percent_to_theta(self, percent):
        """ Convert percent to the angle theta, in radians.

        theta is the angle of the needle measured counterclockwise from
        the horizontal (i.e. the traditional angle of polar coordinates).
        """
        angle = (
            self.angle
            + (180.0 - 2 * self.angle)
            * (self.max_percent - percent)
            / self.max_percent
        )
        theta = math.radians(angle)
        return theta

    def _db_to_theta(self, db):
        """ Convert db to the angle theta, in radians. """
        percent = db_to_percent(db)
        theta = self._percent_to_theta(percent)
        return theta
