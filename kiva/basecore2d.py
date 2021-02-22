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
Pure-Python reference implementation of a Kiva graphics context.

Data Structures
---------------

color
    1-D array with 4 elements.
    The array elements represent (red, green, blue, alpha)
    and are each between 0 and 1.  Alpha is a transparency
    value with 0 being fully transparent and 1 being fully
    opaque.  Many backends do not handle tranparency and
    treat any alpha value greater than 0 as fully opaque.
transform
    currently a 3x3 array.  This is not the
    most convenient in some backends.  Mac and OpenGL
    use a 1-D 6 element array.  We need to either make
    transform a class or always use accessor functions
    to access its values. Currently, I do the latter.

"""
import numpy as np
from numpy import alltrue, array, asarray, concatenate, float64, pi, shape

from .constants import (
    CAP_BUTT, CAP_ROUND, CAP_SQUARE, CLOSE, CONCAT_CTM, EOF_FILL_STROKE,
    EOF_FILL, FILL_STROKE, FILL, JOIN_BEVEL, JOIN_MITER, JOIN_ROUND, LINE,
    LINES, LOAD_CTM, NO_DASH, POINT, RECT, ROTATE_CTM, SCALE_CTM, STROKE,
    TEXT_CLIP, TEXT_FILL_CLIP, TEXT_FILL_STROKE_CLIP, TEXT_FILL_STROKE,
    TEXT_FILL, TEXT_INVISIBLE, TEXT_OUTLINE, TEXT_STROKE_CLIP, TEXT_STROKE,
    TRANSLATE_CTM,
)
from .abstract_graphics_context import AbstractGraphicsContext
from .line_state import LineState, line_state_equal
from .graphics_state import GraphicsState
from .fonttools import Font
import kiva.affine as affine

# --------------------------------------------------------------------
# Drawing style tests.
#
# Simple tests used by drawing methods to determine what kind of
# drawing command is supposed to be executed.
# --------------------------------------------------------------------


def is_point(tup):
    return tup[0] == POINT


def is_line(tup):
    return tup[0] == LINE


def is_fully_transparent(color):
    """ Tests a color array to see whether it is fully transparent or not.

    This is true if the alpha value (4th entry in the color array) is 0.0.
    """
    transparent = color[3] == 0.0
    return transparent


def fill_equal(fill1, fill2):
    """ Compares the two fill colors. """
    return alltrue(fill1 == fill2)


class GraphicsContextBase(AbstractGraphicsContext):
    """ Concrete base implementation of a GraphicsContext

    Attributes
    ----------

    state
        Current state of graphics context.
    state_stack
        Stack used to save graphics states
    path
        The drawing path.
    active_subpath
        The active drawing subpath

    This class needs to be sub-classed by device types that handle
    drawing but don't handle more advanced concepts like paths, graphics state,
    and coordinate transformations.

    This class can also be used as a null backend for testing purposes.
    """

    def __init__(self, *args, **kwargs):
        super(GraphicsContextBase, self).__init__()
        self.state = GraphicsState()

        # The line state has multiple properties that are tracked by a class
        self.last_drawn_line_state = LineState(None, None, None, None, None)

        # The fill state is simply a color.
        self.last_drawn_fill_state = None
        self.last_font_state = None

        # Used by save/restore state.
        self.state_stack = []

        # Variables for used in drawing paths.
        # The path_transform_indices holds a list of indices pointing into
        # active_subpath that affect the ctm.  It is necessary to preserve
        # these across begin_path calls.
        self.active_subpath = []
        self.path_transform_indices = []
        self.path = [self.active_subpath]

        # Whether the particular underlying graphics context considers the
        # "origin" of a pixel to be the center of the pixel or the lower-left
        # corner.  Most vector-based drawing systems consider the origin to
        # be at the corner, whereas most raster systems place the origin at
        # the center.
        #
        # This is ultimately used to determine whether certain methods should
        # automatically tack on a (0.5, 0.5) offset.
        self.corner_pixel_origin = True

        # --------------------------------------------------------------------
        # We're currently maintaining a couple of copies of the ctm around.
        # The state.ctm is used mainly for user querying, etc.  We also have
        # something called the device_ctm which is actually used in the
        # drawing of objects.  In some implementation (OpenGL), the
        # device_ctm is actually maintained in hardware.
        # --------------------------------------------------------------------
        self.device_prepare_device_ctm()

    # ------------------------------------------------------------------------
    # Coordinate Transform Matrix Manipulation
    #
    # Note:  I'm not sure we really need to keep the state.ctm around now
    #        that we're keeping the device_ctm around, but I'm reluctant to
    #        unify the two yet.  I think it can (and probably should) be done
    #        though.
    # ------------------------------------------------------------------------

    def scale_ctm(self, sx, sy):
        """ Sets the coordinate system scale to the given values, (sx, sy).

            Parameters
            ----------
            sx : float
                The new scale factor for the x axis
            sy : float
                The new scale factor for the y axis
        """
        self.state.ctm = affine.scale(self.state.ctm, sx, sy)
        self.active_subpath.append((SCALE_CTM, (sx, sy)))
        self.path_transform_indices.append(len(self.active_subpath) - 1)

    def translate_ctm(self, tx, ty):
        """ Translates the coordinate system by the value given by (tx, ty)

            Parameters
            ----------
            tx : float
                The distance to move in the x direction
            ty : float
                The distance to move in the y direction
        """
        self.state.ctm = affine.translate(self.state.ctm, tx, ty)
        self.active_subpath.append((TRANSLATE_CTM, (tx, ty)))
        self.path_transform_indices.append(len(self.active_subpath) - 1)

    def rotate_ctm(self, angle):
        """ Rotates the coordinate space for drawing by the given angle.

            Parameters
            ----------
            angle : float
                the angle, in radians, to rotate the coordinate system
        """
        self.state.ctm = affine.rotate(self.state.ctm, angle)
        self.active_subpath.append((ROTATE_CTM, (angle,)))
        self.path_transform_indices.append(len(self.active_subpath) - 1)

    def concat_ctm(self, transform):
        """ Concatenates the transform to current coordinate transform matrix.

            Parameters
            ----------
            transform : affine_matrix
                the transform matrix to concatenate with
                the current coordinate matrix.
        """
        self.state.ctm = affine.concat(self.state.ctm, transform)
        self.active_subpath.append((CONCAT_CTM, (transform,)))
        self.path_transform_indices.append(len(self.active_subpath) - 1)

    def get_ctm(self):
        """ Returns the current coordinate transform matrix.
        """
        return self.state.ctm.copy()

    def set_ctm(self, transform):
        """ Returns the current coordinate transform matrix.
        """
        self.state.ctm = transform
        self.active_subpath.append((LOAD_CTM, (transform,)))
        self.path_transform_indices.append(len(self.active_subpath) - 1)

    # ----------------------------------------------------------------
    # Save/Restore graphics state.
    # ----------------------------------------------------------------

    def save_state(self):
        """ Saves the current graphic's context state.

        Always pair this with a `restore_state()`, for example using
        try ... finally ... or the context manager interface.

        """
        self.state_stack.append(self.state)
        self.state = self.state.copy()

    def restore_state(self):
        """ Restores the previous graphics state. """
        self.state = self.state_stack.pop(-1)
        self.active_subpath.append((LOAD_CTM, (self.state.ctm,)))
        self.path_transform_indices.append(len(self.active_subpath) - 1)

    # ----------------------------------------------------------------
    # context manager interface
    # ----------------------------------------------------------------

    def __enter__(self):
        self.save_state()

    def __exit__(self, type, value, traceback):
        self.restore_state()

    # ----------------------------------------------------------------
    # Manipulate graphics state attributes.
    # ----------------------------------------------------------------

    def set_antialias(self, value):
        """ Sets/Unsets anti-aliasing for bitmap graphics context.

        Ignored on most platforms.

        """
        self.state.antialias = value

    def get_antialias(self, value):
        """ Returns the anti-aliasing for bitmap graphics context.

        Ignored on most platforms.

        """
        return self.state.antialias

    def set_image_interpolation(self, value):
        """ Sets image interpolation for bitmap graphics context.

        Ignored on most platforms.

        """
        self.state.image_interpolation = value

    def get_image_interpolation(self, value):
        """ Sets/Unsets anti-aliasing for bitmap graphics context.

        Ignored on most platforms.

        """
        return self.state.image_interpolation

    def set_line_width(self, width):
        """ Sets the line width for drawing

        Parameters
        ----------
        width : float
            The new width for lines in user space units.

        """
        self.state.line_state.line_width = width

    def set_line_join(self, style):
        """ Sets the style for joining lines in a drawing.

        Parameters
        ----------
        style : join_style
            The line joining style.  The available
            styles are JOIN_ROUND, JOIN_BEVEL, JOIN_MITER.

        """
        if style not in (JOIN_ROUND, JOIN_BEVEL, JOIN_MITER):
            msg = "Invalid line join style. See documentation for valid styles"
            raise ValueError(msg)
        self.state.line_state.line_join = style

    def set_miter_limit(self, limit):
        """ Specifies limits on line lengths for mitering line joins.

        If line_join is set to miter joins, the limit specifies which
        line joins should actually be mitered.  If lines are not mitered,
        they are joined with a bevel.  The line width is divided by
        the length of the miter.  If the result is greater than the
        limit, the bevel style is used.

        This is not implemented on most platforms.

        Parameters
        ----------
        limit : float
            limit for mitering joins. defaults to 1.0.

        """
        self.state.miter_limit = limit

    def set_line_cap(self, style):
        """ Specifies the style of endings to put on line ends.

        Parameters
        ----------
        style : cap_style
            The line cap style to use. Available styles
            are CAP_ROUND, CAP_BUTT, CAP_SQUARE.

        """
        if style not in (CAP_ROUND, CAP_BUTT, CAP_SQUARE):
            msg = "Invalid line cap style.  See documentation for valid styles"
            raise ValueError(msg)
        self.state.line_state.line_cap = style

    def set_line_dash(self, pattern, phase=0):
        """ Sets the line dash pattern and phase for line painting.

        Parameters
        ----------
        pattern : float array
            An array of floating point values
            specifing the lengths of on/off painting
            pattern for lines.
        phase : float
            Specifies how many units into dash pattern
            to start.  phase defaults to 0.

        """
        if not alltrue(pattern):
            self.state.line_state.line_dash = NO_DASH
            return
        pattern = asarray(pattern)
        if len(pattern) < 2:
            raise ValueError("dash pattern should have at least two entries.")
        # not sure if this check is really needed.
        if phase < 0:
            raise ValueError("dash phase should be a positive value.")
        self.state.line_state.line_dash = (phase, pattern)

    def set_flatness(self, flatness):
        """ Not implemented

        It is device dependent and therefore not recommended by
        the PDF documentation.

        flatness determines how accurately curves are rendered.  Setting it
        to values less than one will result in more accurate drawings, but
        they take longer.  It defaults to None

        """
        self.state.flatness = flatness

    # ----------------------------------------------------------------
    # Sending drawing data to a device
    # ----------------------------------------------------------------

    def flush(self):
        """ Sends all drawing data to the destination device. """
        pass

    def synchronize(self):
        """ Prepares drawing data to be updated on a destination device.

        Currently this is a NOP for all implementations.

        """
        pass

    # ----------------------------------------------------------------
    # Page Definitions
    # ----------------------------------------------------------------

    def begin_page(self):
        """ Creates a new page within the graphics context.

        Currently this is a NOP for all implementations.  The PDF
        backend should probably implement it, but the ReportLab
        Canvas uses the showPage() method to handle both
        begin_page and end_page issues.
        """
        pass

    def end_page(self):
        """ Ends drawing in the current page of the graphics context.

        Currently this is a NOP for all implementations.  The PDF
        backend should probably implement it, but the ReportLab
        Canvas uses the showPage() method to handle both
        begin_page and end_page issues.
        """
        pass

    # ----------------------------------------------------------------
    # Building paths (contours that are drawn)
    #
    # + Currently, nothing is drawn as the path is built.  Instead, the
    #   instructions are stored and later drawn.  Should this be changed?
    #   We will likely draw to a buffer instead of directly to the canvas
    #   anyway.
    #
    #   Hmmm. No.  We have to keep the path around for storing as a
    #   clipping region and things like that.
    #
    # + I think we should keep the current_path_point hanging around.
    #
    # ----------------------------------------------------------------

    def begin_path(self):
        """ Clears the current drawing path and begin a new one.
        """
        # Need to check here if the current subpath contains matrix
        # transforms.  If  it does, pull these out, and stick them
        # in the new subpath.
        if self.path_transform_indices:
            path_arr = array(self.active_subpath, object)
            tf = path_arr[self.path_transform_indices, :]
            self.path_transform_indices = list(range(len(tf)))
            self.active_subpath = list(tf)
        else:
            self.active_subpath = []
        self.path = [self.active_subpath]

    def move_to(self, x, y):
        """ Starts a new drawing subpath and place the current point at (x, y).

            Notes:
                Not sure how to treat state.current_point.  Should it be the
                value of the point before or after the matrix transformation?
                It looks like before in the PDF specs.
        """
        self._new_subpath()

        pt = array((x, y), dtype=float64)
        self.state.current_point = pt
        self.active_subpath.append((POINT, pt))

    def line_to(self, x, y):
        """ Adds a line from the current point to the given point (x, y).

            The current point is moved to (x, y).

            What should happen if move_to hasn't been called? Should it always
            begin at (0, 0) or raise an error?

            Notes:
                See note in move_to about the current_point.
        """
        pt = array((x, y), dtype=float64)
        self.state.current_point = pt
        self.active_subpath.append((LINE, pt))

    def lines(self, points):
        """ Adds a series of lines as a new subpath.

            Parameters
            ----------

            points
                an Nx2 array of x, y pairs

            The current_point is moved to the last point in 'points'
        """
        self._new_subpath()
        pts = points
        self.active_subpath.append((LINES, pts))
        self.state.current_point = points[-1]

    def line_set(self, starts, ends):
        """ Adds a set of disjoint lines as a new subpath.

            Parameters
            ----------
            starts
                an Nx2 array of x, y pairs
            ends
                an Nx2 array of x, y pairs

            Starts and ends should have the same length.
            The current point is moved to the last point in 'ends'.
        """
        self._new_subpath()
        for i in range(min(len(starts), len(ends))):
            self.active_subpath.append((POINT, starts[i]))
            self.active_subpath.append((LINE, ends[i]))
        self.state.current_point = ends[i]

    def rect(self, x, y, sx, sy):
        """ Adds a rectangle as a new subpath.
        """
        pts = array(((x, y), (x, y + sy), (x + sx, y + sy), (x + sx, y)))
        self.lines(pts)
        self.close_path("rect")

    def draw_rect(self, rect, mode):
        self.rect(*rect)
        self.draw_path(mode=mode)

    def rects(self, rects):
        """ Adds multiple rectangles as separate subpaths to the path.

            Not very efficient -- calls rect multiple times.
        """
        for x, y, sx, sy in rects:
            self.rect(x, y, sx, sy)

    def close_path(self, tag=None):
        """ Closes the path of the current subpath.

            Currently starts a new subpath -- is this what we want?
        """
        self.active_subpath.append((CLOSE, (tag,)))
        self._new_subpath()

    def curve_to(self, x_ctrl1, y_ctrl1, x_ctrl2, y_ctrl2, x_to, y_to):
        """ Draw a cubic bezier curve from the current point.

        Parameters
        ----------
        x_ctrl1 : float
            X-value of the first control point.
        y_ctrl1 : float
            Y-value of the first control point.
        x_ctrl2 : float
            X-value of the second control point.
        y_ctrl2 : float
            Y-value of the second control point.
        x_to : float
            X-value of the ending point of the curve.
        y_to : float
            Y-value of the ending point of the curve.
        """
        # XXX: figure out a reasonable number of points from the current scale
        # and arc length. Since the arc length is expensive to calculate, the
        # sum of the lengths of the line segments from (xy0, xy_ctrl1),
        # (xy_ctrl1, xy_ctrl2), and (xy_ctrl2, xy_to) would be a reasonable
        # approximation.
        n = 100
        t = np.arange(1, n + 1) / float(n)
        t2 = t * t
        t3 = t2 * t
        u = 1 - t
        u2 = u * u
        u3 = u2 * u
        x0, y0 = self.state.current_point
        pts = np.column_stack(
            [
                x0*u3 + 3*(x_ctrl1*t*u2 + x_ctrl2*t2*u) + x_to*t3,
                y0*u3 + 3*(y_ctrl1*t*u2 + y_ctrl2*t2*u) + y_to*t3,
            ]
        )
        self.active_subpath.append((LINES, pts))
        self.state.current_point = pts[-1]

    def quad_curve_to(self, x_ctrl, y_ctrl, x_to, y_to):
        """ Draw a quadratic bezier curve from the current point.

        Parameters
        ----------
        x_ctrl : float
            X-value of the control point
        y_ctrl : float
            Y-value of the control point.
        x_to : float
            X-value of the ending point of the curve
        y_to : float
            Y-value of the ending point of the curve.
        """
        # A quadratic Bezier curve is just a special case of the cubic. Reuse
        # its implementation in case it has been implemented for the specific
        # backend.
        x0, y0 = self.state.current_point
        xc1 = (x0 + x_ctrl + x_ctrl) / 3.0
        yc1 = (y0 + y_ctrl + y_ctrl) / 3.0
        xc2 = (x_to + x_ctrl + x_ctrl) / 3.0
        yc2 = (y_to + y_ctrl + y_ctrl) / 3.0
        self.curve_to(xc1, yc1, xc2, yc2, x_to, y_to)

    def arc(self, x, y, radius, start_angle, end_angle, cw=False):
        """ Draw a circular arc.

        If there is a current path and the current point is not the initial
        point of the arc, a line will be drawn to the start of the arc. If
        there is no current path, then no line will be drawn.

        Parameters
        ----------
        x : float
            X-value of the center of the arc.
        y : float
            Y-value of the center of the arc.
        radius : float
            The radius of the arc.
        start_angle : float
            The angle, in radians, that the starting point makes with respect
            to the positive X-axis from the center point.
        end_angle : float
            The angles, in radians, that the final point makes with
            respect to the positive X-axis from the center point.
        cw : bool, optional
            Whether the arc should be drawn clockwise or not.
        """
        # XXX: pick the number of line segments based on the current scale and
        # the radius.
        n = 100
        if end_angle < start_angle and not cw:
            end_angle += 2 * pi
        elif start_angle < end_angle and cw:
            start_angle += 2 * pi
        theta = np.linspace(start_angle, end_angle, n)
        pts = radius * np.column_stack([np.cos(theta), np.sin(theta)])
        pts += np.array([x, y])
        self.active_subpath.append((LINES, pts))
        self.state.current_point = pts[-1]

    def arc_to(self, x1, y1, x2, y2, radius):
        """
        """
        raise NotImplementedError("arc_to is not implemented")

    def _new_subpath(self):
        """ Starts a new drawing subpath.

            Only creates a new subpath if the current one contains objects.
        """
        if self.active_subpath:
            self.active_subpath = []
            self.path_transform_indices = []
            self.path.append(self.active_subpath)

    # ----------------------------------------------------------------
    # Getting infomration on paths
    # ----------------------------------------------------------------

    def is_path_empty(self):
        """ Tests to see whether the current drawing path is empty
        """
        # If the first subpath is empty, then the path is empty
        res = 0
        if not self.path[0]:
            res = 1
        else:
            res = 1
            for sub in self.path:
                if not is_point(sub[-1]):
                    res = 0
                    break
        return res

    def get_path_current_point(self):
        """ Returns the current point from the graphics context.

            Note:
                Currently the current_point is only affected by move_to,
                line_to, and lines.  It should also be affected by text
                operations.  I'm not sure how rect and rects and friends
                should affect it -- will find out on Mac.
        """
        pass

    def get_path_bounding_box(self):
        """
        """
        pass

    def from_agg_affine(self, aff):
        """Convert an agg.AffineTransform to a numpy matrix
        representing the affine transform usable by kiva.affine
        and other non-agg parts of kiva"""
        return array(
            [
                [aff[0], aff[1], 0],
                [aff[2], aff[3], 0],
                [aff[4], aff[5], 1],
            ],
            float64,
        )

    def add_path(self, path):
        """Draw a compiled path into this gc.  Note: if the CTM is
        changed and not restored to the identity in the compiled path,
        the CTM change will continue in this GC."""
        # Local import to avoid a dependency if we can avoid it.
        from kiva import agg

        multi_state = 0  # For multi-element path commands we keep the previous
        x_ctrl1 = 0  # information in these variables.
        y_ctrl1 = 0
        x_ctrl2 = 0
        y_ctrl2 = 0
        for x, y, cmd, flag in path._vertices():
            if cmd == agg.path_cmd_line_to:
                self.line_to(x, y)
            elif cmd == agg.path_cmd_move_to:
                self.move_to(x, y)
            elif cmd == agg.path_cmd_stop:
                self.concat_ctm(path.get_kiva_ctm())
            elif cmd == agg.path_cmd_end_poly:
                self.close_path()
            elif cmd == agg.path_cmd_curve3:
                if multi_state == 0:
                    x_ctrl1 = x
                    y_ctrl1 = y
                    multi_state = 1
                else:
                    self.quad_curve_to(x_ctrl1, y_ctrl1, x, y)
                    multi_state = 0
            elif cmd == agg.path_cmd_curve4:
                if multi_state == 0:
                    x_ctrl1 = x
                    y_ctrl1 = y
                    multi_state = 1
                elif multi_state == 1:
                    x_ctrl2 = x
                    y_ctrl2 = y
                    multi_state = 2
                elif multi_state == 2:
                    self.curve_to(x_ctrl1, y_ctrl1, x_ctrl2, y_ctrl2, x, y)

    # ----------------------------------------------------------------
    # Clipping path manipulation
    # ----------------------------------------------------------------

    def clip(self):
        """
        """
        pass

    def even_odd_clip(self):
        """
        """
        pass

    def clip_to_rect(self, x, y, width, height):
        """
            Sets the clipping path to the intersection of the current clipping
            path with the area defined by the specified rectangle
        """
        if not self.state.clipping_path:
            self.state.clipping_path = (x, y, width, height)
            self.device_set_clipping_path(x, y, width, height)
        else:
            # Find the intersection of the clipping regions:
            xmin1, ymin1, width1, height1 = self.state.clipping_path
            xclip_min = max(xmin1, x)
            xclip_max = min(xmin1 + width1, x + width)
            yclip_min = max(ymin1, y)
            yclip_max = min(ymin1 + height1, y + height)
            height_clip = max(0, yclip_max - yclip_min)
            width_clip = max(0, xclip_max - xclip_min)
            self.state.clipping_path = (
                xclip_min, yclip_min, width_clip, height_clip,
            )
            self.device_set_clipping_path(
                xclip_min, yclip_min, width_clip, height_clip
            )

    def clip_to_rects(self):
        """
        """
        pass

    def clear_clip_path(self):
        self.state.clipping_path = None
        self.device_destroy_clipping_path()

    # ----------------------------------------------------------------
    # Color manipulation
    # ----------------------------------------------------------------

    def set_fill_color(self, color):
        """
            set_fill_color takes a sequences of rgb or rgba values
            between 0.0 and 1.0
        """
        if len(color) == 3:
            self.state.fill_color[:3] = color
            self.state.fill_color[3] = 1.0
        else:
            self.state.fill_color[:] = color

    def get_fill_color(self, color):
        """
            set_fill_color returns a sequence of rgb or rgba values
            between 0.0 and 1.0
        """
        return self.state.fill_color

    def linear_gradient(self, x1, y1, x2, y2, stops, spread_method, units):
        """ Modify the fill color to be a linear gradient """
        pass

    def radial_gradient(self, cx, cy, r, fx, fy, stops, spread_method, units):
        """ Modify the fill color to be a linear gradient """
        pass

    def set_stroke_color(self, color):
        """
            set_stroke_color takes a sequences of rgb or rgba values
            between 0.0 and 1.0
        """
        if len(color) == 3:
            self.state.line_state.line_color[:3] = color
            self.state.line_state.line_color[3] = 1.0
        else:
            self.state.line_state.line_color[:] = color

    def get_stroke_color(self, color):
        """
            set_stroke_color returns a sequence of rgb or rgba values
            between 0.0 and 1.0
        """
        return self.state.stroke_color

    def set_alpha(self, alpha):
        """ Set the alpha to use when drawing """
        self.state.alpha = alpha

    def get_alpha(self, alpha):
        """ Return the alpha used when drawing """
        return self.state.alpha

    # ----------------------------------------------------------------
    # Drawing Images
    # ----------------------------------------------------------------

    def draw_image(self, img, rect=None):
        """
        """
        self.device_draw_image(img, rect)

    # -------------------------------------------------------------------------
    # Drawing Text
    #
    # Font handling needs more attention.
    #
    # -------------------------------------------------------------------------

    def select_font(self, face_name, size=12, style="regular", encoding=None):
        """ Selects a new font for drawing text.

            Parameters
            ----------

            face_name
                The name of a font. E.g.: "Times New Roman"
                !! Need to specify a way to check for all the types
                size
                The font size in points.
            style
                One of "regular", "bold", "italic", "bold italic"
            encoding
                A 4 letter encoding name. Common ones are:

                    * "unic" -- unicode
                    * "armn" -- apple roman
                    * "symb" -- symbol

                 Not all fonts support all encodings.  If none is
                 specified, fonts that have unicode encodings
                 default to unicode.  Symbol is the second choice.
                 If neither are available, the encoding defaults
                 to the first one returned in the FreeType charmap
                 list for the font face.
        """
        # !! should check if name and encoding are valid.
        self.state.font = Font(face_name, size=size, style=style)

    def set_font(self, font):
        """ Set the font for the current graphics context.
        """
        self.state.font = font.copy()

    def get_font(self, font):
        """ Set the font for the current graphics context.
        """
        return self.state.font.copy()

    def set_font_size(self, size):
        """ Sets the size of the font.

            The size is specified in user space coordinates.

            Note:
                I don't think the units of this are really "user space
                coordinates" on most platforms.  I haven't looked into
                the text drawing that much, so this stuff needs more
                attention.
        """
        self.state.font.size = size

    def set_character_spacing(self, spacing):
        """ Sets the amount of additional spacing between text characters.

            Parameters
            ----------

            spacing : float
                units of space extra space to add between
                text coordinates.  It is specified in text coordinate
                system.

            Notes
            -----
            1.  I'm assuming this is horizontal spacing?
            2.  Not implemented in wxPython.
        """
        self.state.character_spacing = spacing

    def get_character_spacing(self):
        """ Gets the amount of additional spacing between text characters. """
        return self.state.character_spacing

    def set_text_drawing_mode(self, mode):
        """ Specifies whether text is drawn filled or outlined or both.

            Parameters
            ----------

            mode
                determines how text is drawn to the screen.  If
                a CLIP flag is set, the font outline is added to the
                clipping path. Possible values:

                    TEXT_FILL
                        fill the text
                    TEXT_STROKE
                        paint the outline
                    TEXT_FILL_STROKE
                        fill and outline
                    TEXT_INVISIBLE
                        paint it invisibly ??
                    TEXT_FILL_CLIP
                        fill and add outline clipping path
                    TEXT_STROKE_CLIP
                        outline and add outline to clipping path
                    TEXT_FILL_STROKE_CLIP
                        fill, outline, and add to clipping path
                    TEXT_CLIP
                        add text outline to clipping path

            Note:
                wxPython currently ignores all but the INVISIBLE flag.
        """
        text_modes = (
            TEXT_FILL, TEXT_STROKE, TEXT_FILL_STROKE, TEXT_INVISIBLE,
            TEXT_FILL_CLIP, TEXT_STROKE_CLIP, TEXT_FILL_STROKE_CLIP, TEXT_CLIP,
            TEXT_OUTLINE,
        )
        if mode not in text_modes:
            msg = (
                "Invalid text drawing mode.  See documentation for valid "
                + "modes"
            )
            raise ValueError(msg)
        self.state.text_drawing_mode = mode

    def set_text_position(self, x, y):
        """
        """
        a, b, c, d, tx, ty = affine.affine_params(self.state.text_matrix)
        tx, ty = x, y
        self.state.text_matrix = affine.affine_from_values(a, b, c, d, tx, ty)

    def get_text_position(self):
        """
        """
        a, b, c, d, tx, ty = affine.affine_params(self.state.text_matrix)
        return tx, ty

    def set_text_matrix(self, ttm):
        """
        """
        self.state.text_matrix = ttm.copy()

    def get_text_matrix(self):
        """
        """
        return self.state.text_matrix.copy()

    def show_text(self, text):
        """ Draws text on the device at the current text position.

            This calls the device dependent device_show_text() method to
            do all the heavy lifting.

            It is not clear yet how this should affect the current point.
        """
        self.device_show_text(text)

    def show_text_tanslate(self, text, dx, dy):
        """ Draws text at the specified offset. """
        x, y = self.get_text_position()
        self.set_text_position(x + dx, y + dy)
        self.device_show_text(text)
        self.set_text_position(x, y)

    # ------------------------------------------------------------------------
    # kiva defaults to drawing text using the freetype rendering engine.
    #
    # If you would like to use a systems native text rendering engine,
    # override this method in the class concrete derived from this one.
    # ------------------------------------------------------------------------
    def device_show_text(self, text):
        """ Draws text on the device at the current text position.

            This relies on the FreeType engine to render the text to an array
            and then calls the device dependent device_show_text() to display
            the rendered image to the screen.

            !! antiliasing is turned off until we get alpha blending
            !! of images figured out.
        """

        # This is not currently implemented in a device-independent way.

    def show_glyphs(self):
        """
        """
        pass

    def show_text_at_point(self, text, x, y):
        """
        """
        pass

    def show_glyphs_at_point(self):
        """
        """
        pass

    # ----------------------------------------------------------------
    # Painting paths (drawing and filling contours)
    # ----------------------------------------------------------------

    def get_empty_path(self):
        """ Get an empty CompiledPath instance """
        pass

    def stroke_path(self):
        self.draw_path(mode=STROKE)

    def fill_path(self):
        self.draw_path(mode=FILL)

    def eof_fill_path(self):
        self.draw_path(mode=EOF_FILL)

    def draw_path(self, mode=FILL_STROKE):
        """ Walks through all the drawing subpaths and draw each element.

            Each subpath is drawn separately.

            Parameters
            ----------
            mode
                Specifies how the subpaths are drawn.  The default is
                FILL_STROKE.  The following are valid values.

                    FILL
                        Paint the path using the nonzero winding rule
                        to determine the regions for painting.
                    EOF_FILL
                        Paint the path using the even-odd fill rule.
                    STROKE
                        Draw the outline of the path with the
                        current width, end caps, etc settings.
                    FILL_STROKE
                        First fill the path using the nonzero
                        winding rule, then stroke the path.
                    EOF_FILL_STROKE
                        First fill the path using the even-odd
                        fill method, then stroke the path.
        """
        # ---------------------------------------------------------------------
        # FILL AND STROKE settings are handled by setting the alpha value of
        # the line and fill colors to zero (transparent) if stroke or fill
        # is not needed.
        # ---------------------------------------------------------------------

        old_line_alpha = self.state.line_state.line_color[3]
        old_fill_alpha = self.state.fill_color[3]
        if mode not in [STROKE, FILL_STROKE, EOF_FILL_STROKE]:
            self.state.line_state.line_color[3] = 0.0
        if mode not in [FILL, EOF_FILL, FILL_STROKE, EOF_FILL_STROKE]:
            self.state.fill_color[3] = 0.0

        self.device_update_line_state()
        self.device_update_fill_state()

        ctm_funcs = (
            SCALE_CTM, ROTATE_CTM, TRANSLATE_CTM, CONCAT_CTM, LOAD_CTM,
        )
        for subpath in self.path:
            # reset the current point for drawing.
            self.clear_subpath_points()
            for func, args in subpath:
                if func == POINT:
                    self.draw_subpath(mode)
                    self.add_point_to_subpath(args)
                    self.first_point = args
                elif func == LINE:
                    self.add_point_to_subpath(args)
                elif func == LINES:
                    self.draw_subpath(mode)
                    # add all points in list to subpath.
                    self.add_point_to_subpath(args)
                    self.first_point = args[0]
                elif func == CLOSE:
                    self.add_point_to_subpath(self.first_point)
                    self.draw_subpath(mode)
                elif func == RECT:
                    self.draw_subpath(mode)
                    self.device_draw_rect(
                        args[0], args[1], args[2], args[3], mode
                    )
                elif func in ctm_funcs:
                    self.device_transform_device_ctm(func, args)
                else:
                    print("oops:", func)
            # finally, draw any remaining paths.
            self.draw_subpath(mode)

        # ---------------------------------------------------------------------
        # reset the alpha values for line and fill values.
        # ---------------------------------------------------------------------
        self.state.line_state.line_color[3] = old_line_alpha
        self.state.fill_color[3] = old_fill_alpha

        # ---------------------------------------------------------------------
        # drawing methods always consume the path on Mac OS X.  We'll follow
        # this convention to make implementation there easier.
        # ---------------------------------------------------------------------
        self.begin_path()

    def device_prepare_device_ctm(self):
        self.device_ctm = affine.affine_identity()

    def device_transform_device_ctm(self, func, args):
        """ Default implementation for handling scaling matrices.

            Many implementations will just use this function.  Others, like
            OpenGL, can benefit from overriding the method and using
            hardware acceleration.
        """
        if func == SCALE_CTM:
            self.device_ctm = affine.scale(self.device_ctm, args[0], args[1])
        elif func == ROTATE_CTM:
            self.device_ctm = affine.rotate(self.device_ctm, args[0])
        elif func == TRANSLATE_CTM:
            self.device_ctm = affine.translate(
                self.device_ctm, args[0], args[1]
            )
        elif func == CONCAT_CTM:
            self.device_ctm = affine.concat(self.device_ctm, args[0])
        elif func == LOAD_CTM:
            self.device_ctm = args[0].copy()

    def device_draw_rect(self, x, y, sx, sy, mode):
        """ Default implementation of drawing  a rect.
        """
        self._new_subpath()
        # When rectangles are rotated, they have to be drawn as a polygon
        # on most devices.  We'll need to specialize this on API's that
        # can handle rotated rects such as Quartz and OpenGL(?).
        # All transformations are done in the call to lines().
        pts = array(
            ((x, y), (x, y + sy), (x + sx, y + sy), (x + sx, y), (x, y))
        )
        self.add_point_to_subpath(pts)
        self.draw_subpath(mode)

    def stroke_rect(self):
        """
        """
        pass

    def stroke_rect_with_width(self):
        """
        """
        pass

    def fill_rect(self):
        """
        """
        pass

    def fill_rects(self):
        """
        """
        pass

    def clear_rect(self):
        """
        """
        pass

    # ----------------------------------------------------------------
    # Subpath point management and drawing routines.
    # ----------------------------------------------------------------

    def add_point_to_subpath(self, pt):
        self.draw_points.append(pt)

    def clear_subpath_points(self):
        self.draw_points = []

    def get_subpath_points(self, debug=0):
        """ Gets the points that are in the current path.

            The first entry in the draw_points list may actually
            be an array.  If this is true, the other points are
            converted to an array and concatenated with the first
        """
        if self.draw_points and len(shape(self.draw_points[0])) > 1:
            first_points = self.draw_points[0]
            other_points = asarray(self.draw_points[1:])
            if len(other_points):
                pts = concatenate((first_points, other_points), 0)
            else:
                pts = first_points
        else:
            pts = asarray(self.draw_points)
        return pts

    def draw_subpath(self, mode):
        """ Fills and strokes the point path.

            After the path is drawn, the subpath point list is
            cleared and ready for the next subpath.

            Parameters
            ----------

            mode
                Specifies how the subpaths are drawn.  The default is
                FILL_STROKE.  The following are valid values.

                    FILL
                        Paint the path using the nonzero winding rule
                        to determine the regions for painting.
                    EOF_FILL
                        Paint the path using the even-odd fill rule.
                    STROKE
                        Draw the outline of the path with the
                        current width, end caps, etc settings.
                    FILL_STROKE
                        First fill the path using the nonzero
                        winding rule, then stroke the path.
                    EOF_FILL_STROKE
                        First fill the path using the even-odd
                        fill method, then stroke the path.

            Note:
                If path is closed, it is about 50% faster to call
                DrawPolygon with the correct pen set than it is to
                call DrawPolygon and DrawLines separately in wxPython. But,
                because paths can be left open, Polygon can't be
                called in the general case because it automatically
                closes the path.  We might want a separate check in here
                and allow devices to specify a faster version if the path is
                closed.
        """
        pts = self.get_subpath_points()
        if len(pts) > 1:
            self.device_fill_points(pts, mode)
            self.device_stroke_points(pts, mode)
        self.clear_subpath_points()

    def get_text_extent(self, textstring):
        """
            Calls device specific text extent method.
        """
        return self.device_get_text_extent(textstring)

    def device_get_text_extent(self, textstring):
        return self.device_get_full_text_extent(textstring)

    def get_full_text_extent(self, textstring):
        """
            Calls device specific text extent method.
        """
        return self.device_get_full_text_extent(textstring)

    def device_get_full_text_extent(self, textstring):
        return (0.0, 0.0, 0.0, 0.0)

    # -------------------------------------------
    # Misc functions
    # -------------------------------------------

    def save(self, filename, file_format=None, pil_options=None):
        """ Save the graphics context to a file """
        raise NotImplementedError
