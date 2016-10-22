# -----------------------------------------------------------------------------
# Copyright (c) 2016, Enthought, Inc
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in enthought/LICENSE.txt and may be redistributed only
# under the conditions described in the aforementioned license.  The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
# Thanks for using Enthought open source!
# -----------------------------------------------------------------------------
from __future__ import absolute_import, print_function, division

from collections import namedtuple
import warnings

import numpy as np
import pyagg as agg

from .abstract_graphics_context import AbstractGraphicsContext
from .fonttools import Font
import kiva.constants as constants

# These are the symbols that a backend has to define.
__all__ = ["CompiledPath", "Font", "font_metrics_provider", "GraphicsContext"]

cap_style = {
    constants.CAP_ROUND: agg.LineCap.CapRound,
    constants.CAP_SQUARE: agg.LineCap.CapSquare,
    constants.CAP_BUTT: agg.LineCap.CapButt,
}
join_style = {
    constants.JOIN_ROUND: agg.LineJoin.JoinRound,
    constants.JOIN_BEVEL: agg.LineJoin.JoinBevel,
    constants.JOIN_MITER: agg.LineJoin.JoinMiter,
}
draw_modes = {
    constants.FILL: agg.DrawingMode.DrawFill,
    constants.EOF_FILL: agg.DrawingMode.DrawEofFill,
    constants.STROKE: agg.DrawingMode.DrawStroke,
    constants.FILL_STROKE: agg.DrawingMode.DrawFillStroke,
    constants.EOF_FILL_STROKE: agg.DrawingMode.DrawEofFillStroke,
}
text_modes = {
    constants.TEXT_FILL: agg.TextDrawingMode.TextDrawFill,
    constants.TEXT_STROKE: agg.TextDrawingMode.TextDrawStroke,
    constants.TEXT_FILL_STROKE: agg.TextDrawingMode.TextDrawFillStroke,
    constants.TEXT_INVISIBLE: agg.TextDrawingMode.TextDrawInvisible,
    constants.TEXT_FILL_CLIP: agg.TextDrawingMode.TextDrawFillClip,
    constants.TEXT_STROKE_CLIP: agg.TextDrawingMode.TextDrawStrokeClip,
    constants.TEXT_FILL_STROKE_CLIP: agg.TextDrawingMode.TextDrawFillStrokeClip,  # noqa
    constants.TEXT_CLIP: agg.TextDrawingMode.TextDrawClip,
    constants.TEXT_OUTLINE: agg.TextDrawingMode.TextDrawStroke,
}
gradient_coord_modes = {
    'userSpaceOnUse': agg.GradientUnits.UserSpace,
    'objectBoundingBox': agg.GradientUnits.ObjectBoundingBox,
}
gradient_spread_modes = {
    'pad': agg.GradientSpread.SpreadPad,
    'repeat': agg.GradientSpread.SpreadRepeat,
    'reflect': agg.GradientSpread.SpreadReflect,
}
StateBundle = namedtuple('StateBundle',
                         ['state', 'path', 'stroke', 'fill', 'transform',
                          'text_transform', 'font'])


class GraphicsContext(object):
    def __init__(self, size, *args, **kwargs):
        super(GraphicsContext, self).__init__()
        self._width = size[0]
        self._height = size[1]

        shape = (self._height, self._width, 4)
        self.gc = agg.CanvasBGRA32(np.zeros(shape, dtype=np.uint8),
                                   bottom_up=True)

        # init the state variables
        self.canvas_state = agg.GraphicsState()
        self.stroke_paint = agg.SolidPaint(0.0, 0.0, 0.0)
        self.fill_paint = agg.SolidPaint(0.0, 0.0, 0.0)
        self.path = CompiledPath()
        self.text_transform = agg.Transform()
        self.text_pos = (0.0, 0.0)
        self.transform = agg.Transform()
        self.font = None
        self.__state_stack = []

    # ----------------------------------------------------------------
    # Size info
    # ----------------------------------------------------------------

    def height(self):
        """ Returns the height of the context.
        """
        return self._height

    def width(self):
        """ Returns the width of the context.
        """
        return self._width

    # ----------------------------------------------------------------
    # Coordinate Transform Matrix Manipulation
    # ----------------------------------------------------------------

    def scale_ctm(self, sx, sy):
        """ Set the coordinate system scale to the given values, (sx, sy).

            sx:float -- The new scale factor for the x axis
            sy:float -- The new scale factor for the y axis
        """
        self.transform.scale(sx, sy)

    def translate_ctm(self, tx, ty):
        """ Translate the coordinate system by the given value by (tx, ty)

            tx:float --  The distance to move in the x direction
            ty:float --   The distance to move in the y direction
        """
        self.transform.translate(tx, ty)

    def rotate_ctm(self, angle):
        """ Rotates the coordinate space for drawing by the given angle.

            angle:float -- the angle, in radians, to rotate the coordinate
                           system
        """
        self.transform.rotate(angle)

    def concat_ctm(self, transform):
        """ Concatenate the transform to current coordinate transform matrix.

            transform:affine_matrix -- the transform matrix to concatenate with
                                       the current coordinate matrix.
        """
        self.transform.premultiply(agg.Transform(*transform))

    def get_ctm(self):
        """ Return the current coordinate transform matrix.
        """
        t = self.transform
        return (t.sx, t.shy, t.shx, t.sy, t.tx, t.ty)

    # ----------------------------------------------------------------
    # Save/Restore graphics state.
    # ----------------------------------------------------------------

    def save_state(self):
        """ Save the current graphic's context state.

            This should always be paired with a restore_state
        """
        state = StateBundle(
            state=self.canvas_state.copy(),
            path=self.path.copy(),
            stroke=self.stroke_paint.copy(),
            fill=self.fill_paint.copy(),
            transform=self.transform.copy(),
            text_transform=self.text_transform.copy(),
            font=(None if self.font is None else self.font.copy()))
        self.__state_stack.append(state)

    def restore_state(self):
        """ Restore the previous graphics state.
        """
        state = self.__state_stack.pop()
        self.canvas_state = state.state
        self.path = state.path
        self.stroke_paint = state.stroke
        self.fill_paint = state.fill
        self.transform = state.transform
        self.text_transform = state.text_transform
        self.font = state.font

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
        """ Set/Unset antialiasing for bitmap graphics context.
        """
        self.canvas_state.anti_aliased = value

    def set_line_width(self, width):
        """ Set the line width for drawing

            width:float -- The new width for lines in user space units.
        """
        self.canvas_state.line_width = width

    def set_line_join(self, style):
        """ Set style for joining lines in a drawing.

            style:join_style -- The line joining style.  The available
                                styles are JOIN_ROUND, JOIN_BEVEL, JOIN_MITER.
        """
        try:
            sjoin = join_style[style]
            self.canvas_state.line_join = sjoin
        except KeyError:
            msg = "Invalid line join style. See documentation for valid styles"
            raise ValueError(msg)

    def set_miter_limit(self, limit):
        """ Specifies limits on line lengths for mitering line joins.

            If line_join is set to miter joins, the limit specifies which
            line joins should actually be mitered.  If lines aren't mitered,
            they are joined with a bevel.  The line width is divided by
            the length of the miter.  If the result is greater than the
            limit, the bevel style is used.

            limit:float -- limit for mitering joins.
        """
        self.canvas_state.miter_limit = limit

    def set_line_cap(self, style):
        """ Specify the style of endings to put on line ends.

            style:cap_style -- the line cap style to use. Available styles
                               are CAP_ROUND, CAP_BUTT, CAP_SQUARE
        """
        try:
            scap = cap_style[style]
            self.canvas_state.line_cap = scap
        except KeyError:
            msg = "Invalid line cap style.  See documentation for valid styles"
            raise ValueError(msg)

    def set_line_dash(self, lengths, phase=0):
        """

            lengths:float array -- An array of floating point values
                                   specifing the lengths of on/off painting
                                   pattern for lines.
            phase:float -- Specifies how many units into dash pattern
                           to start.  phase defaults to 0.
        """
        if lengths:
            count = len(lengths)
            lengths = np.array(lengths).reshape(count//2, 2)
        else:
            lengths = []
        self.canvas_state.line_dash_pattern = lengths
        self.canvas_state.line_dash_phase = phase

    def set_flatness(self, flatness):
        """ Not implemented

            It is device dependent and therefore not recommended by
            the PDF documentation.
        """
        msg = "set_flatness not implemented for pyagg"
        raise NotImplementedError(msg)

    # ----------------------------------------------------------------
    # Sending drawing data to a device
    # ----------------------------------------------------------------

    def flush(self):
        """ Send all drawing data to the destination device.
        """

    def synchronize(self):
        """ Prepares drawing data to be updated on a destination device.
        """

    # ----------------------------------------------------------------
    # Page Definitions
    # ----------------------------------------------------------------

    def begin_page(self):
        """ Create a new page within the graphics context.
        """

    def end_page(self):
        """ End drawing in the current page of the graphics context.
        """

    # ----------------------------------------------------------------
    # Path creation
    # ----------------------------------------------------------------

    def begin_path(self):
        """ Clear the current drawing path and begin a new one.
        """
        self.path.path.reset()

    def move_to(self, x, y):
        """ Start a new drawing subpath at place the current point at (x, y).
        """
        self.path.move_to(x, y)

    def line_to(self, x, y):
        """ Add a line from the current point to the given point (x, y).

            The current point is moved to (x, y).
        """
        self.path.line_to(x, y)

    def lines(self, points):
        """ Add a series of lines as a new subpath.

            Currently implemented by calling line_to a zillion times.

            Points is an Nx2 array of x, y pairs.
        """
        self.path.lines(points)

    def line_set(self, starts, ends):
        """ Draw multiple disjoint line segments.
        """
        self.path.line_set(starts, ends)

    def rect(self, x, y, sx, sy):
        """ Add a rectangle as a new subpath.
        """
        self.path.rect(x, y, sx, sy)

    def rects(self, rects):
        """ Add multiple rectangles as separate subpaths to the path.
        """
        self.path.rects(rects)

    def draw_rect(self, rect, mode=constants.FILL_STROKE):
        """ Draw a rect.
        """
        path = agg.Path()
        path.rect(*rect)

        self.canvas_state.drawing_mode = draw_modes[mode]
        self.gc.draw_shape(path, self.transform, self.stroke_paint,
                           self.fill_paint, self.canvas_state)

    def add_path(self, path):
        """ Add a subpath to the current path.
        """
        self.path.add_path(path)

    def close_path(self):
        """ Close the path of the current subpath.
        """
        self.path.close_path()

    def curve_to(self, cp1x, cp1y, cp2x, cp2y, x, y):
        self.path.curve_to(cp1x, cp1y, cp2x, cp2y, x, y)

    def quad_curve_to(self, cpx, cpy, x, y):
        self.path.quad_curve_to(cpx, cpy, x, y)

    def arc(self, x, y, radius, start_angle, end_angle, clockwise=False):
        self.path.arc(x, y, radius, start_angle, end_angle, clockwise)

    def arc_to(self, x1, y1, x2, y2, radius):
        self.path.arc_to(x1, y1, x2, y2, radius)

    # ----------------------------------------------------------------
    # Getting infomration on paths
    # ----------------------------------------------------------------

    def is_path_empty(self):
        """ Test to see if the current drawing path is empty
        """
        return self.path.is_empty()

    def get_path_current_point(self):
        """ Return the current point from the graphics context.
        """
        return self.path.get_current_point()

    def get_path_bounding_box(self):
        """ Return the bounding box for the current path object.
        """
        return self.path.get_bounding_box()

    # ----------------------------------------------------------------
    # Clipping path manipulation
    # ----------------------------------------------------------------

    def clip(self):
        """ Clip context to a filled version of the current path.
        """
        if not self.path.is_empty():
            self._clip_impl(self.path.path, agg.DrawingMode.DrawFill)
            self.begin_path()

    def even_odd_clip(self):
        """ Clip context to a even-odd filled version of the current path.
        """
        if not self.path.is_empty():
            self._clip_impl(self.path.path, agg.DrawingMode.DrawEofFill)
            self.begin_path()

    def clip_to_rect(self, x, y, w, h):
        """ Clip context to the given rectangular region.

            Region should be a 4-tuple or a sequence.
        """
        self.canvas_state.clip_box = agg.Rect(x, y, x + w, y + h)

    def clip_to_rects(self, rects):
        """ Clip context to a collection of rectangles
        """
        path = agg.Path()
        path.rects(rects)
        self._clip_impl(path, agg.DrawingMode.DrawFill)

    def _clip_impl(self, shape, drawing_mode):
        """ Internal implementation for the complex clipping methods.
        """
        size = (self._height, self._width)
        stencil = agg.CanvasG8(np.empty(size, dtype=np.uint8), bottom_up=True)
        stencil.clear(0.0, 0.0, 0.0)

        gs = agg.GraphicsState(drawing_mode=drawing_mode)
        paint = agg.SolidPaint(1.0, 1.0, 1.0)
        stencil.draw_shape(shape, self.transform, paint, paint, gs)

        self.canvas_state.stencil = stencil.image

    # ----------------------------------------------------------------
    # Color space manipulation
    #
    # I'm not sure we'll mess with these at all.  They seem to
    # be for setting the color system.  Hard coding to RGB or
    # RGBA for now sounds like a reasonable solution.
    # ----------------------------------------------------------------

    def set_fill_color_space(self):
        msg = "set_fill_color_space not implemented for pyagg yet."
        raise NotImplementedError(msg)

    def set_stroke_color_space(self):
        msg = "set_stroke_color_space not implemented for pyagg yet."
        raise NotImplementedError(msg)

    def set_rendering_intent(self):
        msg = "set_rendering_intent not implemented for pyagg yet."
        raise NotImplementedError(msg)

    # ----------------------------------------------------------------
    # Color manipulation
    # ----------------------------------------------------------------

    def set_fill_color(self, color):
        self.fill_paint = agg.SolidPaint(*color)

    def set_stroke_color(self, color):
        self.stroke_paint = agg.SolidPaint(*color)

    def set_alpha(self, alpha):
        self.canvas_state.master_alpha = alpha

    # ----------------------------------------------------------------
    # Gradients
    # ----------------------------------------------------------------

    def _get_gradient_enums(self, spread, units):
        """ Configures a gradient object and sets it as the current brush.
        """
        spread = gradient_spread_modes.get(spread,
                                           agg.GradientSpread.SpreadPad)
        units = gradient_coord_modes.get(units, agg.GradientUnits.UserSpace)
        return spread, units

    def linear_gradient(self, x1, y1, x2, y2, stops, spread_method,
                        units='userSpaceOnUse'):
        """ Sets a linear gradient as the current brush.
        """
        spread, units = self._get_gradient_enums(spread_method, units)
        self.fill_paint = agg.LinearGradientPaint(x1, y1, x2, y2,
                                                  stops, spread, units)

    def radial_gradient(self, cx, cy, r, fx, fy, stops, spread_method,
                        units='userSpaceOnUse'):
        """ Sets a radial gradient as the current brush.
        """
        spread, units = self._get_gradient_enums(spread_method, units)
        self.fill_paint = agg.RadialGradientPaint(cx, cy, r, fx, fy,
                                                  stops, spread, units)

    # ----------------------------------------------------------------
    # Drawing Images
    # ----------------------------------------------------------------

    def draw_image(self, img, rect=None):
        """
        img is either a N*M*3 or N*M*4 numpy array, or a Kiva image

        rect - a tuple (x, y, w, h)
        """
        img_format = agg.PixelFormat.RGB24
        if isinstance(img, np.ndarray):
            # Numeric array
            img_array = img.astype(np.uint8)
            if img.shape[2] == 3:
                img_format = agg.PixelFormat.RGB24
            elif img.shape[2] == 4:
                img_format = agg.PixelFormat.RGBA32
        elif isinstance(img, agg.CanvasBGRA32):
            img_array = img.array
            img_format = agg.PixelFormat.BGRA32
        elif hasattr(img, 'bmp_array'):
            # An offscreen kiva context
            # XXX: Use a copy to kill the read-only flag which plays havoc
            # with the Cython memoryviews used by pyagg
            img_array = img.bmp_array.copy()
            img_format = agg.PixelFormat.RGB24
        else:
            msg = "Cannot render image of type '{}' into pyagg context."
            warnings.warn(msg.format(type(img)))
            return

        # Calculate the scale to use
        rw, rh = rect[2:]
        ih, iw = img_array.shape[:2]
        sx, sy = rw / iw, rh / ih

        transform = agg.Transform()
        transform.multiply(self.transform)
        if sx != 0.0 or sy != 0.0:
            transform.scale(sx, sy)
        transform.translate(*rect[:2])

        self.gc.draw_image(img_array, img_format, transform, self.canvas_state,
                           bottom_up=True)

    # ----------------------------------------------------------------
    # Drawing Text
    # ----------------------------------------------------------------

    def select_font(self, name, size, textEncoding):
        """ Set the font for the current graphics context.
        """
        self.font = agg.Font(name, size, agg.FontCacheType.VectorFontCache)

    def set_font(self, font):
        """ Set the font for the current graphics context.
        """
        self.select_font(font.findfont(), font.size, None)

    def set_font_size(self, size):
        msg = "set_font_size not implemented on pyagg yet."
        raise NotImplementedError(msg)

    def set_character_spacing(self, spacing):
        msg = "set_character_spacing not implemented on pyagg yet."
        raise NotImplementedError(msg)

    def set_text_drawing_mode(self, mode):
        try:
            tmode = text_modes[mode]
        except KeyError:
            msg = "Invalid text drawing mode"
            raise ValueError(msg)
        self.canvas_state.text_drawing_mode = tmode

    def set_text_position(self, x, y):
        self.text_pos = (x, y)

    def get_text_position(self):
        return self.text_pos

    def set_text_matrix(self, ttm):
        self.text_transform = ttm

    def get_text_matrix(self):
        return self.text_transform

    def show_text(self, text, point=None):
        """ Draw text on the device at current text position.

            This is also used for showing text at a particular point
            specified by x and y.
        """
        if point is None:
            pos = tuple(self.text_pos)
        else:
            pos = tuple(point)

        transform = agg.Transform()
        transform.multiply(self.transform)
        transform.translate(*pos)

        self.gc.draw_text(text, self.font, transform, self.stroke_paint,
                          self.stroke_paint, self.canvas_state)

    def show_text_at_point(self, text, x, y):
        """ Draw text at some point (x, y).
        """
        self.show_text(text, (x, y))

    def show_glyphs(self):
        msg = "show_glyphs not implemented on pyagg"
        raise NotImplementedError(msg)

    def get_text_extent(self, text):
        """ Returns the bounding rect of the rendered text
        """
        x, y = self.transform.tx, self.transform.ty
        width = self.font.width(text)
        height = self.font.height
        return x, y, x + width, y + height

    def get_full_text_extent(self, text):
        """ Backwards compatibility API over .get_text_extent() for Enable
        """
        x1, y1, x2, y2 = self.get_text_extent(text)

        return x2, y2, y1, x1

    # ----------------------------------------------------------------
    # Painting paths (drawing and filling contours)
    # ----------------------------------------------------------------

    def stroke_path(self):
        self.canvas_state.drawing_mode = agg.DrawingMode.DrawStroke
        self.gc.draw_shape(self.path.path, self.transform, self.stroke_paint,
                           self.fill_paint, self.canvas_state)
        self.begin_path()

    def fill_path(self):
        self.canvas_state.drawing_mode = agg.DrawingMode.DrawFill
        self.gc.draw_shape(self.path.path, self.transform, self.stroke_paint,
                           self.fill_paint, self.canvas_state)
        self.begin_path()

    def eof_fill_path(self):
        self.canvas_state.drawing_mode = agg.DrawingMode.DrawEofFill
        self.gc.draw_shape(self.path.path, self.transform, self.stroke_paint,
                           self.fill_paint, self.canvas_state)
        self.begin_path()

    def stroke_rect(self, rect):
        self.stroke_rect_with_width(rect, 1.0)

    def stroke_rect_with_width(self, rect, width):
        shape = agg.Path()
        shape.rect(*rect)

        self.canvas_state.line_width = width
        self.canvas_state.drawing_mode = agg.DrawingMode.DrawStroke
        self.gc.draw_shape(shape, self.transform, self.stroke_paint,
                           self.fill_paint, self.canvas_state)

    def fill_rect(self, rect):
        shape = agg.Path()
        shape.rect(*rect)

        self.canvas_state.drawing_mode = agg.DrawingMode.DrawFill
        self.gc.draw_shape(shape, self.transform, self.stroke_paint,
                           self.fill_paint, self.canvas_state)

    def fill_rects(self):
        msg = "fill_rects not implemented on pyagg yet."
        raise NotImplementedError(msg)

    def clear_rect(self, rect):
        msg = "clear_rect not implemented on pyagg yet."
        raise NotImplementedError(msg)

    def clear(self, clear_color=(1.0, 1.0, 1.0, 1.0)):
        self.gc.clear(*clear_color)

    def draw_path(self, mode=constants.FILL_STROKE):
        """ Walk through all the drawing subpaths and draw each element.

            Each subpath is drawn separately.
        """
        self.canvas_state.drawing_mode = draw_modes[mode]
        self.gc.draw_shape(self.path.path, self.transform, self.stroke_paint,
                           self.fill_paint, self.canvas_state)
        self.begin_path()

    def get_empty_path(self):
        """ Return a path object that can be built up and then reused.
        """
        return CompiledPath()

    def draw_path_at_points(self, points, path, mode=constants.FILL_STROKE):
        """ Draw a path object at many different points.

        XXX: This is currently broken for some reason
        """
        # set up drawing state and function
        shape = agg.ShapeAtPoints(self.path.path, points)

        self.canvas_state.drawing_mode = draw_modes[mode]
        self.gc.draw_shape(shape, self.transform, self.stroke_paint,
                           self.fill_paint, self.canvas_state)

    def save(self, filename, file_format=None):
        """ Save the contents of the context to a file
        """
        msg = "save not implemented on pyagg yet."
        raise NotImplementedError(msg)


class CompiledPath(object):
    def __init__(self):
        self.path = agg.Path()

    def copy(self):
        cpy = CompiledPath()
        cpy.path = self.path.copy()
        return cpy

    def begin_path(self):
        self.path.begin()

    def move_to(self, x, y):
        self.path.move_to(x, y)

    def arc(self, x, y, r, start_angle, end_angle, clockwise=False):
        self.path.arc(x, y, r, start_angle, end_angle, clockwise)

    def arc_to(self, x1, y1, x2, y2, r):
        self.path.arc_to(x1, y1, x2, y2, r)

    def line_to(self, x, y):
        self.path.line_to(x, y)

    def lines(self, points):
        self.path.lines(points)

    def line_set(self, starts, ends):
        self.path.lines_set(starts, ends)

    def curve_to(self, cx1, cy1, cx2, cy2, x, y):
        self.path.cubic_to(cx1, cy1, cx2, cy2, x, y)

    def quad_curve_to(self, cx, cy, x, y):
        self.path.quadric_to(cx, cy, x, y)

    def rect(self, x, y, sx, sy):
        self.path.rect(x, y, sx, sy)

    def rects(self, rects):
        self.path.rects(rects)

    def add_path(self, other_path):
        if isinstance(other_path, CompiledPath):
            self.path.addPath(other_path.path)

    def close_path(self):
        self.path.close()

    def is_empty(self):
        return self.path.length() == 0

    def get_current_point(self):
        return self.path.final_point()

    def get_bounding_box(self):
        return self.path.bounding_rect()


# GraphicsContext should implement AbstractGraphicsContext
AbstractGraphicsContext.register(GraphicsContext)


def font_metrics_provider():
    """ Creates an object to be used for querying font metrics.
    """
    return GraphicsContext((1, 1))
