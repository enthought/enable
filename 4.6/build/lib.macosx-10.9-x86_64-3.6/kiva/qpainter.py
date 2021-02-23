# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
""" This is the QPainter backend for kiva. """

from functools import partial
import numpy as np
import warnings

# Major package imports.
from pyface.qt import QtCore, QtGui

# Local imports.
from .abstract_graphics_context import AbstractGraphicsContext
from .arc_conversion import arc_to_tangent_points
from .fonttools import Font
import kiva.constants as constants

# These are the symbols that a backend has to define.
__all__ = ["CompiledPath", "Font", "font_metrics_provider", "GraphicsContext"]

cap_style = {}
cap_style[constants.CAP_ROUND] = QtCore.Qt.RoundCap
cap_style[constants.CAP_SQUARE] = QtCore.Qt.SquareCap
cap_style[constants.CAP_BUTT] = QtCore.Qt.FlatCap

join_style = {}
join_style[constants.JOIN_ROUND] = QtCore.Qt.RoundJoin
join_style[constants.JOIN_BEVEL] = QtCore.Qt.BevelJoin
join_style[constants.JOIN_MITER] = QtCore.Qt.MiterJoin

draw_modes = {}
draw_modes[constants.FILL] = QtCore.Qt.OddEvenFill
draw_modes[constants.EOF_FILL] = QtCore.Qt.WindingFill
draw_modes[constants.STROKE] = 0
draw_modes[constants.FILL_STROKE] = QtCore.Qt.OddEvenFill
draw_modes[constants.EOF_FILL_STROKE] = QtCore.Qt.WindingFill

gradient_coord_modes = {}
gradient_coord_modes["userSpaceOnUse"] = QtGui.QGradient.LogicalMode
gradient_coord_modes["objectBoundingBox"] = QtGui.QGradient.ObjectBoundingMode

gradient_spread_modes = {}
gradient_spread_modes["pad"] = QtGui.QGradient.PadSpread
gradient_spread_modes["repeat"] = QtGui.QGradient.RepeatSpread
gradient_spread_modes["reflect"] = QtGui.QGradient.ReflectSpread


class GraphicsContext(object):
    """ Simple wrapper around a Qt QPainter object.
    """

    def __init__(self, size, *args, **kwargs):
        super(GraphicsContext, self).__init__()
        self._width = size[0]
        self._height = size[1]

        self.text_pos = [0.0, 0.0]
        self.text_transform = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)

        # create some sort of device context
        parent = kwargs.pop("parent", None)
        if parent is None:
            # no parent -> offscreen context
            self.qt_dc = QtGui.QPixmap(*size)
        else:
            # normal windowed context
            self.qt_dc = parent

        self.gc = QtGui.QPainter(self.qt_dc)
        self.path = CompiledPath()

        # For HiDPI support, we only need to adjust for `size`
        base_pixel_scale = kwargs.pop("base_pixel_scale", 1)

        # flip y
        trans = QtGui.QTransform()
        trans.translate(0, size[1] / base_pixel_scale)
        trans.scale(1.0, -1.0)
        self.gc.setWorldTransform(trans)

        # enable antialiasing
        self.gc.setRenderHints(
            QtGui.QPainter.Antialiasing | QtGui.QPainter.TextAntialiasing, True
        )
        # set the pen and brush to useful defaults
        self.gc.setPen(QtCore.Qt.black)
        self.gc.setBrush(QtGui.QBrush(QtCore.Qt.SolidPattern))

    def __del__(self):
        # stop the painter if needed
        if self.gc.isActive():
            self.gc.end()

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
        self.gc.scale(sx, sy)

    def translate_ctm(self, tx, ty):
        """ Translate the coordinate system by the given value by (tx, ty)

            tx:float --  The distance to move in the x direction
            ty:float --   The distance to move in the y direction
        """
        self.gc.translate(tx, ty)

    def rotate_ctm(self, angle):
        """ Rotates the coordinate space for drawing by the given angle.

            angle:float -- the angle, in radians, to rotate the coordinate
                           system
        """
        self.gc.rotate(np.rad2deg(angle))

    def concat_ctm(self, transform):
        """ Concatenate the transform to current coordinate transform matrix.

            transform:affine_matrix -- the transform matrix to concatenate with
                                       the current coordinate matrix.
        """
        m11, m12, m21, m22, tx, ty = transform
        self.gc.setTransform(
            QtGui.QTransform(m11, m12, m21, m22, tx, ty), True
        )

    def get_ctm(self):
        """ Return the current coordinate transform matrix.
        """
        t = self.gc.transform()
        return (t.m11(), t.m12(), t.m21(), t.m22(), t.dx(), t.dy())

    # ----------------------------------------------------------------
    # Save/Restore graphics state.
    # ----------------------------------------------------------------

    def save_state(self):
        """ Save the current graphic's context state.

            This should always be paired with a restore_state
        """
        self.gc.save()

    def restore_state(self):
        """ Restore the previous graphics state.
        """
        self.gc.restore()

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
        self.gc.setRenderHints(
            QtGui.QPainter.Antialiasing | QtGui.QPainter.TextAntialiasing,
            value,
        )

    def set_line_width(self, width):
        """ Set the line width for drawing

            width:float -- The new width for lines in user space units.
        """
        pen = self.gc.pen()
        pen.setWidthF(width)
        self.gc.setPen(pen)

    def set_line_join(self, style):
        """ Set style for joining lines in a drawing.

            style:join_style -- The line joining style.  The available
                                styles are JOIN_ROUND, JOIN_BEVEL, JOIN_MITER.
        """
        try:
            sjoin = join_style[style]
        except KeyError:
            msg = "Invalid line join style. See documentation for valid styles"
            raise ValueError(msg)

        pen = self.gc.pen()
        pen.setJoinStyle(sjoin)
        self.gc.setPen(pen)

    def set_miter_limit(self, limit):
        """ Specifies limits on line lengths for mitering line joins.

            If line_join is set to miter joins, the limit specifies which
            line joins should actually be mitered.  If lines aren't mitered,
            they are joined with a bevel.  The line width is divided by
            the length of the miter.  If the result is greater than the
            limit, the bevel style is used.

            limit:float -- limit for mitering joins.
        """
        pen = self.gc.pen()
        pen.setMiterLimit(limit)
        self.gc.setPen(pen)

    def set_line_cap(self, style):
        """ Specify the style of endings to put on line ends.

            style:cap_style -- the line cap style to use. Available styles
                               are CAP_ROUND, CAP_BUTT, CAP_SQUARE
        """
        try:
            scap = cap_style[style]
        except KeyError:
            msg = "Invalid line cap style.  See documentation for valid styles"
            raise ValueError(msg)

        pen = self.gc.pen()
        pen.setCapStyle(scap)
        self.gc.setPen(pen)

    def set_line_dash(self, lengths, phase=0):
        """

            lengths:float array -- An array of floating point values
                                   specifing the lengths of on/off painting
                                   pattern for lines.
            phase:float -- Specifies how many units into dash pattern
                           to start.  phase defaults to 0.
        """
        lengths = list(lengths) if lengths is not None else []
        pen = self.gc.pen()
        pen.setDashPattern(lengths)
        pen.setDashOffset(phase)
        self.gc.setPen(pen)

    def set_flatness(self, flatness):
        """ Not implemented

            It is device dependent and therefore not recommended by
            the PDF documentation.
        """
        raise NotImplementedError()

    # ----------------------------------------------------------------
    # Sending drawing data to a device
    # ----------------------------------------------------------------

    def flush(self):
        """ Send all drawing data to the destination device.
        """
        pass

    def synchronize(self):
        """ Prepares drawing data to be updated on a destination device.
        """
        pass

    # ----------------------------------------------------------------
    # Page Definitions
    # ----------------------------------------------------------------

    def begin_page(self):
        """ Create a new page within the graphics context.
        """
        pass

    def end_page(self):
        """ End drawing in the current page of the graphics context.
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
        """ Clear the current drawing path and begin a new one.
        """
        self.path = CompiledPath()

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
        for start, end in zip(starts, ends):
            self.path.path.moveTo(start[0], start[1])
            self.path.path.lineTo(end[0], end[1])

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
        rect = QtCore.QRectF(*rect)
        if mode == constants.STROKE:
            save_brush = self.gc.brush()
            self.gc.setBrush(QtGui.QBrush(QtCore.Qt.NoBrush))
            self.gc.drawRect(rect)
            self.gc.setBrush(save_brush)
        elif mode in [constants.FILL, constants.EOF_FILL]:
            self.gc.fillRect(rect, self.gc.brush())
        else:
            self.gc.fillRect(rect, self.gc.brush())
            self.gc.drawRect(rect)

    def add_path(self, path):
        """ Add a subpath to the current path.
        """
        self.path.add_path(path)

    def close_path(self):
        """ Close the path of the current subpath.
        """
        self.path.close_path()

    def curve_to(self, cp1x, cp1y, cp2x, cp2y, x, y):
        """
        """
        self.path.curve_to(cp1x, cp1y, cp2x, cp2y, x, y)

    def quad_curve_to(self, cpx, cpy, x, y):
        """
        """
        self.path.quad_curve_to(cpx, cpy, x, y)

    def arc(self, x, y, radius, start_angle, end_angle, clockwise=False):
        """
        """
        self.path.arc(x, y, radius, start_angle, end_angle, clockwise)

    def arc_to(self, x1, y1, x2, y2, radius):
        """
        """
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
        """
        """
        self.gc.setClipPath(self.path.path)

    def even_odd_clip(self):
        """
        """
        self.gc.setClipPath(self.path.path, operation=QtCore.Qt.IntersectClip)

    def clip_to_rect(self, x, y, w, h):
        """ Clip context to the given rectangular region.

            Region should be a 4-tuple or a sequence.
        """
        self.gc.setClipRect(
            QtCore.QRectF(x, y, w, h), operation=QtCore.Qt.IntersectClip
        )

    def clip_to_rects(self, rects):
        """
        """
        # Create a region which is a union of all rects.
        clip_region = QtGui.QRegion()
        for rect in rects:
            clip_region = clip_region.unite(QtGui.QRegion(*rect))

        # Then intersect that region with the current clip region.
        self.gc.setClipRegion(clip_region, operation=QtCore.Qt.IntersectClip)

    # ----------------------------------------------------------------
    # Color space manipulation
    #
    # I'm not sure we'll mess with these at all.  They seem to
    # be for setting the color system.  Hard coding to RGB or
    # RGBA for now sounds like a reasonable solution.
    # ----------------------------------------------------------------

    def set_fill_color_space(self):
        """
        """
        msg = "set_fill_color_space not implemented on Qt yet."
        raise NotImplementedError(msg)

    def set_stroke_color_space(self):
        """
        """
        msg = "set_stroke_color_space not implemented on Qt yet."
        raise NotImplementedError(msg)

    def set_rendering_intent(self):
        """
        """
        msg = "set_rendering_intent not implemented on Qt yet."
        raise NotImplementedError(msg)

    # ----------------------------------------------------------------
    # Color manipulation
    # ----------------------------------------------------------------

    def set_fill_color(self, color):
        """
        """
        r, g, b = color[:3]
        try:
            a = color[3]
        except IndexError:
            a = 1.0
        brush = self.gc.brush()
        brush.setColor(QtGui.QColor.fromRgbF(r, g, b, a))
        self.gc.setBrush(brush)

    def set_stroke_color(self, color):
        """
        """
        r, g, b = color[:3]
        try:
            a = color[3]
        except IndexError:
            a = 1.0
        pen = self.gc.pen()
        pen.setColor(QtGui.QColor.fromRgbF(r, g, b, a))
        self.gc.setPen(pen)

    def set_alpha(self, alpha):
        """
        """
        self.gc.setOpacity(alpha)

    # ----------------------------------------------------------------
    # Gradients
    # ----------------------------------------------------------------

    def _apply_gradient(self, grad, stops, spread_method, units):
        """ Configures a gradient object and sets it as the current brush.
        """
        grad.setSpread(
            gradient_spread_modes.get(spread_method, QtGui.QGradient.PadSpread)
        )
        grad.setCoordinateMode(
            gradient_coord_modes.get(units, QtGui.QGradient.LogicalMode)
        )

        for stop in stops:
            grad.setColorAt(stop[0], QtGui.QColor.fromRgbF(*stop[1:]))

        self.gc.setBrush(QtGui.QBrush(grad))

    def linear_gradient(self, x1, y1, x2, y2, stops, spread_method,
                        units="userSpaceOnUse"):
        """ Sets a linear gradient as the current brush.
        """
        grad = QtGui.QLinearGradient(x1, y1, x2, y2)
        self._apply_gradient(grad, stops, spread_method, units)

    def radial_gradient(self, cx, cy, r, fx, fy, stops, spread_method,
                        units="userSpaceOnUse"):
        """ Sets a radial gradient as the current brush.
        """
        grad = QtGui.QRadialGradient(cx, cy, r, fx, fy)
        self._apply_gradient(grad, stops, spread_method, units)

    # ----------------------------------------------------------------
    # Drawing Images
    # ----------------------------------------------------------------

    def draw_image(self, img, rect=None):
        """
        img is either a N*M*3 or N*M*4 numpy array, or a PIL Image

        rect - a tuple (x, y, w, h)
        """
        from PIL import Image, ImageQt

        if isinstance(img, np.ndarray):
            # Numeric array
            pilimg = Image.fromarray(img)
            width, height = pilimg.width, pilimg.height
            draw_img = ImageQt.ImageQt(pilimg)
            pixmap = QtGui.QPixmap.fromImage(draw_img)
        elif isinstance(img, Image.Image):
            width, height = img.width, img.height
            draw_img = ImageQt.ImageQt(img)
            pixmap = QtGui.QPixmap.fromImage(draw_img)
        elif hasattr(img, "bmp_array"):
            # An offscreen kiva agg context
            pilimg = Image.fromarray(img.bmp_array)
            width, height = pilimg.width, pilimg.height
            draw_img = ImageQt.ImageQt(pilimg)
            pixmap = QtGui.QPixmap.fromImage(draw_img)
        elif (isinstance(img, GraphicsContext)
                and isinstance(img.qt_dc, QtGui.QPixmap)
                and img.gc.isActive()):
            # An offscreen Qt kiva context
            # Calling qpainter.device() appears to introduce a memory leak.
            # using the display context and calling qpainter.isActive() has
            # the same outcome.
            pixmap = img.qt_dc
            width, height = pixmap.width(), pixmap.height()
        else:
            msg = "Cannot render image of type '%r' into Qt4 context."
            warnings.warn(msg % type(img))
            return

        # create a rect object to draw into
        if rect is None:
            dest_rect = QtCore.QRectF(0.0, 0.0, self.width(), self.height())
        else:
            dest_rect = QtCore.QRectF(*rect)

        # draw using the entire image's data
        source_rect = QtCore.QRectF(0.0, 0.0, width, height)

        flip_trans = QtGui.QTransform()
        flip_trans.scale(1.0, -1.0)
        pixmap = pixmap.transformed(flip_trans)

        # draw
        self.gc.drawPixmap(dest_rect, pixmap, source_rect)

    # ----------------------------------------------------------------
    # Drawing Text
    # ----------------------------------------------------------------

    def select_font(self, name, size, textEncoding):
        """ Set the font for the current graphics context.
        """
        self.gc.setFont(QtGui.QFont(name, size))

    def set_font(self, font):
        """ Set the font for the current graphics context.
        """
        self.select_font(font.face_name, font.size, None)

    def set_font_size(self, size):
        """
        """
        font = self.gc.font()
        font.setPointSizeF(size)
        self.gc.setFont(font)

    def set_character_spacing(self, spacing):
        """
        """
        font = self.gc.font()
        font.setLetterSpacing(QtGui.QFont.AbsoluteSpacing, spacing)
        self.gc.setFont(font)

    def set_text_drawing_mode(self):
        """
        """
        pass

    def set_text_position(self, x, y):
        """
        """
        self.text_pos = [x, y]

    def get_text_position(self):
        """
        """
        return self.text_pos

    def set_text_matrix(self, ttm):
        """
        """
        self.text_transform = ttm

    def get_text_matrix(self):
        """
        """
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

        unflip_trans = QtGui.QTransform(*self.text_transform)
        unflip_trans.translate(0, self._height)
        unflip_trans.scale(1.0, -1.0)

        self.gc.save()
        self.gc.setTransform(unflip_trans, True)
        self.gc.drawText(QtCore.QPointF(pos[0], self._flip_y(pos[1])), text)
        self.gc.restore()

    def show_text_at_point(self, text, x, y):
        """ Draw text at some point (x, y).
        """
        self.show_text(text, (x, y))

    def show_glyphs(self):
        """
        """
        msg = "show_glyphs not implemented on Qt yet."
        raise NotImplementedError(msg)

    def get_text_extent(self, text):
        """ Returns the bounding rect of the rendered text
        """
        fm = self.gc.fontMetrics()
        rect = fm.boundingRect(text)

        return rect.left(), -fm.descent(), rect.right(), fm.height()

    def get_full_text_extent(self, text):
        """ Backwards compatibility API over .get_text_extent() for Enable
        """
        x1, y1, x2, y2 = self.get_text_extent(text)

        return x2, y2, y1, x1

    # ----------------------------------------------------------------
    # Painting paths (drawing and filling contours)
    # ----------------------------------------------------------------

    def stroke_path(self):
        """
        """
        self.gc.strokePath(self.path.path, self.gc.pen())
        self.begin_path()

    def fill_path(self):
        """
        """
        self.gc.fillPath(self.path.path, self.gc.brush())
        self.begin_path()

    def eof_fill_path(self):
        """
        """
        self.path.path.setFillRule(QtCore.Qt.OddEvenFill)
        self.gc.fillPath(self.path.path, self.gc.brush())
        self.begin_path()

    def stroke_rect(self, rect):
        """
        """
        self.gc.drawRect(QtCore.QRectF(*rect))

    def stroke_rect_with_width(self, rect, width):
        """
        """
        save_pen = self.gc.pen()
        draw_pen = QtGui.QPen(save_pen)
        draw_pen.setWidthF(width)

        self.gc.setPen(draw_pen)
        self.stroke_rect(rect)
        self.gc.setPen(save_pen)

    def fill_rect(self, rect):
        """
        """
        self.gc.fillRect(QtCore.QRectF(*rect), self.gc.brush())

    def fill_rects(self):
        """
        """
        msg = "fill_rects not implemented on Qt yet."
        raise NotImplementedError(msg)

    def clear_rect(self, rect):
        """
        """
        self.gc.eraseRect(QtCore.QRectF(*rect))

    def clear(self, clear_color=(1.0, 1.0, 1.0, 1.0)):
        """
        """
        if len(clear_color) == 4:
            r, g, b, a = clear_color
        else:
            r, g, b = clear_color
            a = 1.0
        self.gc.setBackground(QtGui.QBrush(QtGui.QColor.fromRgbF(r, g, b, a)))
        self.gc.eraseRect(QtCore.QRectF(0, 0, self.width(), self.height()))

    def draw_path(self, mode=constants.FILL_STROKE):
        """ Walk through all the drawing subpaths and draw each element.

            Each subpath is drawn separately.
        """
        if mode == constants.STROKE:
            self.stroke_path()
        elif mode in [constants.FILL, constants.EOF_FILL]:
            mode = draw_modes[mode]
            self.path.path.setFillRule(mode)
            self.fill_path()
        else:
            mode = draw_modes[mode]
            self.path.path.setFillRule(mode)
            self.gc.drawPath(self.path.path)
        self.begin_path()

    def get_empty_path(self):
        """ Return a path object that can be built up and then reused.
        """
        return CompiledPath()

    def draw_path_at_points(self, points, path, mode=constants.FILL_STROKE):
        # set up drawing state and function
        if mode == constants.STROKE:
            draw_func = partial(self.gc.strokePath, path.path, self.gc.pen())
        elif mode in [constants.FILL, constants.EOF_FILL]:
            mode = draw_modes[mode]
            path.path.setFillRule(mode)
            draw_func = partial(self.gc.fillPath, path.path, self.gc.brush())
        else:
            mode = draw_modes[mode]
            path.path.setFillRule(mode)
            draw_func = partial(self.gc.drawPath, path.path)

        for point in points:
            x, y = point
            self.gc.save()
            self.gc.translate(x, y)
            draw_func()
            self.gc.restore()

    def _flip_y(self, y):
        "Converts between a Kiva and a Qt y coordinate"
        return self._height - y - 1

    def save(self, filename, file_format=None):
        """ Save the contents of the context to a file
        """
        if isinstance(self.qt_dc, QtGui.QPixmap):
            self.qt_dc.save(filename, format=file_format)
        else:
            msg = "save not implemented for window contexts."
            raise NotImplementedError(msg)


class CompiledPath(object):
    def __init__(self):
        self.path = QtGui.QPainterPath()

    def begin_path(self):
        return

    def move_to(self, x, y):
        self.path.moveTo(x, y)

    def arc(self, x, y, r, start_angle, end_angle, clockwise=False):
        sweep_angle = (
            end_angle - start_angle
            if not clockwise
            else start_angle - end_angle
        )
        self.path.moveTo(x, y)
        self.path.arcTo(
            QtCore.QRectF(x - r, y - r, r * 2, r * 2),
            np.rad2deg(start_angle),
            np.rad2deg(sweep_angle),
        )

    def arc_to(self, x1, y1, x2, y2, r):
        # get the current pen position
        current_point = self.get_current_point()

        # Get the two points on the curve where it touches the line segments
        t1, t2 = arc_to_tangent_points(current_point, (x1, y1), (x2, y2), r)

        # draw!
        self.path.lineTo(*t1)
        self.path.quadTo(x1, y1, *t2)
        self.path.lineTo(x2, y2)

    def line_to(self, x, y):
        self.path.lineTo(x, y)

    def lines(self, points):
        self.path.moveTo(points[0][0], points[0][1])
        for x, y in points[1:]:
            self.path.lineTo(x, y)

    def curve_to(self, cx1, cy1, cx2, cy2, x, y):
        self.path.cubicTo(cx1, cy1, cx2, cy2, x, y)

    def quad_curve_to(self, cx, cy, x, y):
        self.path.quadTo(cx, cy, x, y)

    def rect(self, x, y, sx, sy):
        self.path.addRect(x, y, sx, sy)

    def rects(self, rects):
        for x, y, sx, sy in rects:
            self.path.addRect(x, y, sx, sy)

    def add_path(self, other_path):
        if isinstance(other_path, CompiledPath):
            self.path.addPath(other_path.path)

    def close_path(self):
        self.path.closeSubpath()

    def is_empty(self):
        return self.path.isEmpty()

    def get_current_point(self):
        point = self.path.currentPosition()
        return point.x(), point.y()

    def get_bounding_box(self):
        rect = self.path.boundingRect()
        return rect.x(), rect.y(), rect.width(), rect.height()


# GraphicsContext should implement AbstractGraphicsContext
AbstractGraphicsContext.register(GraphicsContext)


def font_metrics_provider():
    """ Creates an object to be used for querying font metrics.
    """
    return GraphicsContext((1, 1))
