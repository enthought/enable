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
import os
import warnings

import blend2d
import numpy as np

from kiva.abstract_graphics_context import AbstractGraphicsContext
import kiva.constants as constants
from kiva.fonttools import Font

# These are the symbols that a backend has to define.
__all__ = ["CompiledPath", "Font", "font_metrics_provider", "GraphicsContext"]

cap_style = {
    constants.CAP_BUTT: blend2d.StrokeCap.CAP_BUTT,
    constants.CAP_ROUND: blend2d.StrokeCap.CAP_ROUND,
    constants.CAP_SQUARE: blend2d.StrokeCap.CAP_SQUARE,
}
join_style = {
    constants.JOIN_ROUND: blend2d.StrokeJoin.JOIN_ROUND,
    constants.JOIN_BEVEL: blend2d.StrokeJoin.JOIN_BEVEL,
    constants.JOIN_MITER: blend2d.StrokeJoin.JOIN_MITER_BEVEL,
}
gradient_spread_modes = {
    "pad": blend2d.ExtendMode.PAD,
    "repeat": blend2d.ExtendMode.REPEAT,
    "reflect": blend2d.ExtendMode.REFLECT,
}
pix_formats = {
    "gray8": blend2d.Format.A8,
    "rgba32": blend2d.Format.XRGB32,
}


class GraphicsContext(object):
    def __init__(self, size, *args, **kwargs):
        super().__init__()
        self._width = size[0]
        self._height = size[1]
        self.pix_format = kwargs.get("pix_format", "rgba32")

        shape = (self._height, self._width, 4)
        buffer = np.zeros(shape, dtype=np.uint8)
        self._buffer = buffer
        self._image = blend2d.Image(buffer)
        self.gc = blend2d.Context(self._image)

        # Graphics state
        self.path = blend2d.Path()
        self.font = None
        self._kiva_font = None
        self.text_pos = (0, 0)
        self.text_drawing_mode = constants.TEXT_FILL

        # flip y / HiDPI
        self.base_scale = kwargs.pop("base_pixel_scale", 1)
        self.gc.translate(0, size[1])
        self.gc.scale(self.base_scale, -self.base_scale)
        # Lock it in
        self.gc.user_to_meta()

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
        """ Concatenate a scaling to the current transformation matrix
        """
        self.gc.scale(sx, sy)

    def translate_ctm(self, tx, ty):
        """ Concatenate a translation to the current transformation matrix
        """
        self.gc.translate(tx, ty)

    def rotate_ctm(self, angle):
        """ Concatenate a rotation to the current transformation matrix.
        """
        self.gc.rotate(angle)

    def concat_ctm(self, transform):
        """ Concatenate an arbitrary affine matrix to the current
        transformation matrix.
        """
        raise NotImplementedError()

    def get_ctm(self):
        """ Return the current coordinate transform matrix.
        """
        # XXX: Not a useful return value
        return self.gc.user_matrix()

    # ----------------------------------------------------------------
    # Save/Restore graphics state.
    # ----------------------------------------------------------------

    def save_state(self):
        """ Save the current graphics context's state.

        This should always be paired with a restore_state
        """
        # XXX: This doesn't save the current font or path!
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
        raise NotImplementedError()

    def set_line_width(self, width):
        """ Set the width of the pen used to stroke a path """
        self.gc.set_stroke_width(width)

    def set_line_join(self, style):
        """ Set the style of join to use a path corners
        """
        try:
            sjoin = join_style[style]
            self.gc.set_stroke_join(sjoin)
        except KeyError:
            msg = "Invalid line join style. See documentation for valid styles"
            raise ValueError(msg)

    def set_miter_limit(self, limit):
        """ Set the limit at which mitered joins are flattened.

        Only applicable when the line join type is set to ``JOIN_MITER``.
        """
        self.gc.set_stroke_miter_limit(limit)

    def set_line_cap(self, style):
        """ Set the style of cap to use a path ends
        """
        try:
            scap = cap_style[style]
            self.gc.set_stroke_caps(scap)
        except KeyError:
            msg = "Invalid line cap style.  See documentation for valid styles"
            raise ValueError(msg)

    def set_line_dash(self, lengths, phase=0):
        """ Set the dash style to use when stroking a path
        """
        raise NotImplementedError()

    def set_flatness(self, flatness):
        """ Set the error tolerance when drawing curved paths
        """
        msg = "set_flatness not implemented for blend2d"
        raise NotImplementedError(msg)

    # ----------------------------------------------------------------
    # Sending drawing data to a device
    # ----------------------------------------------------------------

    def flush(self):
        """ Send all drawing data to the destination device.
        """
        self.gc.flush()

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
        self.path.clear()

    def move_to(self, x, y):
        """ Start a new drawing subpath at place the current point at (x, y).
        """
        self.path.move_to(x, y)

    def line_to(self, x, y):
        """ Add a line from the current point to (x, y) to the path
        """
        self.path.line_to(x, y)

    def lines(self, points):
        """ Adds a series of lines as a new subpath.
        """
        for (x, y) in points:
            self.path.line_to(x, y)

    def line_set(self, starts, ends):
        """ Draw multiple disjoint line segments.
        """
        for (x1, y1), (x2, y2) in zip(starts, ends):
            self.path.move_to(x1, y1)
            self.path.line_to(x2, y2)

    def rect(self, x, y, sx, sy):
        """ Add a rectangle as a new subpath.
        """
        self.path.add_rect(x, y, sx, sy)

    def rects(self, rects):
        """ Add multiple rectangles as separate subpaths to the path.
        """
        for rect in rects:
            self.path.add_rect(rect)

    def draw_rect(self, rect, mode=constants.FILL_STROKE):
        """ Draw a rect.
        """
        rect = blend2d.Rect(*rect)
        if mode in (constants.FILL, constants.FILL_STROKE):
            self.gc.fill_rect(rect)
        if mode in (constants.STROKE, constants.FILL_STROKE):
            self.gc.stroke_rect(rect)

    def add_path(self, path):
        """ Add a subpath to the current path.
        """
        self.path.add_path(path)

    def close_path(self):
        """ Close the path of the current subpath.
        """
        self.path.close()

    def curve_to(self, cp1x, cp1y, cp2x, cp2y, x, y):
        """ Draw a cubic bezier curve
        """
        self.path.cubic_to(cp1x, cp1y, cp2x, cp2y, x, y)

    def quad_curve_to(self, cpx, cpy, x, y):
        """ Draw a quadratic bezier curve
        """
        self.path.quadric_to(cpx, cpy, x, y)

    def arc(self, x, y, radius, start_angle, end_angle, cw=False):
        """ Draw a circular arc of the given radius, centered at ``(x, y)``
        """
        self.path.arc_to(
            x, y, radius, radius,
            start_angle, math.fabs(end_angle-start_angle),
            forceMoveTo=True
        )

    def arc_to(self, x1, y1, x2, y2, radius):
        """ Draw a circular arc from current point to tangent line
        """
        self.path.arc_quadrant_to(x1, y1, x2, y2)

    # ----------------------------------------------------------------
    # Getting information on paths
    # ----------------------------------------------------------------

    def is_path_empty(self):
        """ Test to see if the current drawing path is empty
        """
        return self.path.empty()

    def get_path_current_point(self):
        """ Return the current point from the graphics context.
        """
        return self.path.get_last_vertex()

    def get_path_bounding_box(self):
        """ Return the bounding box for the current path object.
        """
        # XXX: Returns a blend2d.Rect which is sort of useless...
        self.path.get_bounding_box()


    # ----------------------------------------------------------------
    # Clipping path manipulation
    # ----------------------------------------------------------------

    def clip(self):
        """ Clip context to a filled version of the current path.
        """
        raise NotImplementedError()

    def even_odd_clip(self):
        """ Clip context to a even-odd filled version of the current path.
        """
        raise NotImplementedError()

    def clip_to_rect(self, x, y, w, h):
        """ Clip context to the given rectangular region.

        Region should be a 4-tuple or a sequence.
        """
        self.gc.clip_to_rect(blend2d.Rect(x, y, w, h))

    def clip_to_rects(self, rects):
        """ Clip context to a collection of rectangles
        """
        raise NotImplementedError()

    # ----------------------------------------------------------------
    # Color space manipulation
    #
    # I'm not sure we'll mess with these at all.  They seem to
    # be for setting the color system.  Hard coding to RGB or
    # RGBA for now sounds like a reasonable solution.
    # ----------------------------------------------------------------

    def set_fill_color_space(self):
        msg = "set_fill_color_space not implemented for blend2d yet."
        raise NotImplementedError(msg)

    def set_stroke_color_space(self):
        msg = "set_stroke_color_space not implemented for blend2d yet."
        raise NotImplementedError(msg)

    def set_rendering_intent(self):
        msg = "set_rendering_intent not implemented for blend2d yet."
        raise NotImplementedError(msg)

    # ----------------------------------------------------------------
    # Color manipulation
    # ----------------------------------------------------------------

    def set_fill_color(self, color):
        """ Set the color used to fill the region bounded by a path or when
        drawing text.
        """
        self.gc.set_fill_style(color)

    def set_stroke_color(self, color):
        """ Set the color used when stroking a path
        """
        self.gc.set_stroke_style(color)

    def set_alpha(self, alpha):
        """ Set the alpha to use when drawing """
        self.gc.set_alpha(alpha)

    # ----------------------------------------------------------------
    # Gradients
    # ----------------------------------------------------------------

    def linear_gradient(self, x1, y1, x2, y2, stops, spread_method,
                        units="userSpaceOnUse"):
        """ Sets a linear gradient as the current brush.
        """
        gradient = blend2d.LinearGradient(x1, y1, x2, y2)
        gradient.extend_mode = gradient_spread_modes.get(
            spread_method, blend2d.ExtendMode.PAD
        )
        for stop in stops:
            gradient.add_stop(stop[0], stop[1:])
        self.gc.set_fill_style(gradient)

    def radial_gradient(self, cx, cy, r, fx, fy, stops, spread_method,
                        units="userSpaceOnUse"):
        """ Sets a radial gradient as the current brush.
        """
        gradient = blend2d.RadialGradient(cx, cy, fx, fy, r)
        gradient.extend_mode = gradient_spread_modes.get(
            spread_method, blend2d.ExtendMode.PAD
        )
        for stop in stops:
            gradient.add_stop(stop[0], stop[1:])
        self.gc.set_fill_style(gradient)

    # ----------------------------------------------------------------
    # Drawing Images
    # ----------------------------------------------------------------

    def draw_image(self, img, rect=None):
        """ Render an image into a rectangle
        """
        from PIL import Image

        def normalize_image(img):
            if not img.mode.startswith("RGB"):
                img = img.convert("RGB")
            return img

        if isinstance(img, np.ndarray):
            # Numeric array
            img = Image.fromarray(img)
            img = normalize_image(img)
            img_array = np.array(img)
        elif isinstance(img, Image.Image):
            img = normalize_image(img)
            img_array = np.array(img)
        elif isinstance(img, GraphicsContext):
            img_array = img._buffer
        elif hasattr(img, "bmp_array"):
            # An offscreen kiva.agg context
            # XXX: Use a copy to kill the read-only flag which plays havoc
            # with the Cython memoryviews used by blend2d
            img = Image.fromarray(img.bmp_array)
            img = normalize_image(img)
            img_array = np.array(img)
        else:
            msg = "Cannot render image of type '{}' into blend2d context."
            warnings.warn(msg.format(type(img)))
            return

        # XXX: Upside down!
        dst_rect = blend2d.Rect(*rect)
        w, h = img.width, img.height
        image = blend2d.Image(img_array)
        rect = blend2d.Rect(0, 0, w, h)
        self.gc.blit_scaled_image(dst_rect, image, rect)

    # ----------------------------------------------------------------
    # Drawing Text
    # ----------------------------------------------------------------

    def select_font(self, face_name, size=12, style="regular", encoding=None):
        """ Set the font for the current graphics context.
        """
        self.set_font(Font(face_name, size=size, style=style))

    def set_font(self, font):
        """ Set the font for the current graphics context.
        """
        spec = font.findfont()
        self._kiva_font = font
        self.font = blend2d.Font(spec.filename, font.size)

    def set_font_size(self, size):
        """ Set the font size for the current graphics context.
        """
        if self._kiva_font is None:
            return

        self._kiva_font.size = size
        self.set_font(self._kiva_font)

    def set_character_spacing(self, spacing):
        """ Set the spacing between characters when drawing text
        """
        msg = "set_character_spacing not implemented on blend2d yet."
        raise NotImplementedError(msg)

    def get_character_spacing(self):
        """ Get the current spacing between characters when drawing text """
        msg = "get_character_spacing not implemented on blend2d yet."
        raise NotImplementedError(msg)

    def set_text_drawing_mode(self, mode):
        """ Set the drawing mode to use with text
        """
        supported_modes = {
            constants.TEXT_FILL,
            constants.TEXT_STROKE,
            constants.TEXT_FILL_STROKE,
            constants.TEXT_INVISIBLE,
        }
        if mode not in supported_modes:
            raise NotImplementedError()

        self.text_drawing_mode = mode

    def set_text_position(self, x, y):
        """ Set the current point for drawing text
        """
        self.text_pos = (x, y)

    def get_text_position(self):
        """ Get the current point where text will be drawn """
        return self.text_pos

    def set_text_matrix(self, ttm):
        """ Set the transformation matrix to use when drawing text """
        raise NotImplementedError()

    def get_text_matrix(self):
        """ Get the transformation matrix to use when drawing text """
        raise NotImplementedError()

    def show_text(self, text, point=None):
        """ Draw the specified string at the current point
        """
        if self.font is None:
            raise RuntimeError("show_text called before setting a font!")

        if self.text_drawing_mode == constants.TEXT_INVISIBLE:
            # XXX: This is probably more sophisticated in practice
            return

        # Convert between a Kiva and a Blend2D Y coordinate
        flip_y = (lambda y: self._height - y - 1)

        if point is None:
            pos = tuple(self.text_pos)
        else:
            pos = tuple(point)

        mode = self.text_drawing_mode
        with self.gc:
            self.gc.translate(0, self._height)
            self.gc.scale(1.0, -1.0)
            pos = (pos[0], flip_y(pos[1]))
            if mode in (constants.TEXT_FILL, constants.TEXT_FILL_STROKE):
                self.gc.fill_text(pos, self.font, text)
            if mode in (constants.TEXT_STROKE, constants.TEXT_FILL_STROKE):
                self.gc.stroke_text(pos, self.font, text)

    def show_text_at_point(self, text, x, y):
        """ Draw text at some point (x, y).
        """
        self.show_text(text, (x, y))

    def show_glyphs(self):
        msg = "show_glyphs not implemented on blend2d"
        raise NotImplementedError(msg)

    def get_text_extent(self, text):
        """ Returns the bounding rect of the rendered text
        """
        if self.font is None:
            raise RuntimeError("get_text_extent called before setting a font!")

        raise NotImplementedError()

    def get_full_text_extent(self, text):
        """ Backwards compatibility API over .get_text_extent() for Enable
        """
        raise NotImplementedError()

    # ----------------------------------------------------------------
    # Painting paths (drawing and filling contours)
    # ----------------------------------------------------------------

    def stroke_path(self):
        """ Stroke the current path with pen settings from current state
        """
        self.gc.stroke_path(self.path)
        self.begin_path()

    def fill_path(self):
        """ Fill the current path with fill settings from the current state
        """
        self.gc.fill_path(self.path)
        self.begin_path()

    def eof_fill_path(self):
        """ Fill the current path with fill settings from the current state
        """
        # XXX: Not fully implemented
        # self.gc.set_fill_rule()
        self.gc.fill_path(self.path)
        self.begin_path()

    def clear_rect(self, rect):
        raise NotImplementedError()

    def clear(self, clear_color=(1.0, 1.0, 1.0, 1.0)):
        with self.gc:
            self.gc.set_fill_style(clear_color)
            self.gc.fill_all()

    def draw_path(self, mode=constants.FILL_STROKE):
        """ Draw the current path with the specified mode
        """
        if mode in (constants.FILL, constants.FILL_STROKE):
            self.gc.fill_path(self.path)
        if mode in (constants.STROKE, constants.FILL_STROKE):
            self.gc.stroke_path(self.path)
        self.begin_path()

    def get_empty_path(self):
        """ Return a path object that can be built up and then reused.
        """
        return CompiledPath()

    def draw_path_at_points(self, points, path, mode=constants.FILL_STROKE):
        """ Draw a path object at many different points.
        """
        raise NotImplementedError()

    def draw_marker_at_points(self, points_array, size,
                              marker=constants.SQUARE_MARKER):
        """ Draw a marker at a collection of points
        """
        raise NotImplementedError()

    def save(self, filename, file_format=None, pil_options=None):
        """ Save the contents of the context to a file
        """
        if file_format is None:
            file_format = ""
        if pil_options is None:
            pil_options = {}

        img = self.to_image()

        ext = (
            os.path.splitext(filename)[1][1:] if isinstance(filename, str)
            else ""
        )

        # Check the output format to see if it can handle an alpha channel.
        no_alpha_formats = ("jpg", "bmp", "eps", "jpeg")
        if ext in no_alpha_formats or file_format.lower() in no_alpha_formats:
            img = img.convert("RGB")

        # Check the output format to see if it can handle DPI
        dpi_formats = ("jpg", "png", "tiff", "jpeg")
        if ext in dpi_formats or file_format.lower() in dpi_formats:
            # Assume 72dpi is 1x
            dpi = int(72 * self.base_scale)
            pil_options["dpi"] = (dpi, dpi)

        img.save(filename, format=file_format, **pil_options)

    def to_image(self):
        """ Return the contents of the context as a PIL Image.

        If the graphics context is in BGRA format, it will convert it to
        RGBA for the image.

        Returns
        -------
        img : Image
            A PIL/Pillow Image object with the data in RGBA format.
        """
        try:
            from PIL import Image
        except ImportError:
            raise ImportError("need Pillow to save images")

        # Data is BGRA; Convert to RGBA
        data = np.empty(self._buffer.shape, dtype=np.uint8)
        data[..., 0] = self._buffer[..., 2]
        data[..., 1] = self._buffer[..., 1]
        data[..., 2] = self._buffer[..., 0]
        data[..., 3] = self._buffer[..., 3]

        return Image.fromarray(data, "RGBA")


class CompiledPath(object):
    def __init__(self):
        self.path = blend2d.Path()

    def copy(self):
        raise NotImplementedError()

    def begin_path(self):
        self.path.clear()

    def move_to(self, x, y):
        self.path.move_to(x, y)

    def arc(self, x, y, r, start_angle, end_angle, cw=False):
        self.path.arc_to(
            x, y, r, r,
            start_angle, math.fabs(end_angle-start_angle),
            forceMoveTo=True
        )

    def arc_to(self, x1, y1, x2, y2, r):
        raise NotImplementedError()

    def line_to(self, x, y):
        self.path.line_to(x, y)

    def lines(self, points):
        for (x, y) in points:
            self.path.line_to(x, y)

    def line_set(self, starts, ends):
        for (x1, y1), (x2, y2) in zip(starts, ends):
            self.path.move_to(x1, y1)
            self.path.line_to(x2, y2)

    def curve_to(self, cx1, cy1, cx2, cy2, x, y):
        self.path.cubic_to(cx1, cy1, cx2, cy2, x, y)

    def quad_curve_to(self, cx, cy, x, y):
        self.path.quadric_to(cx, cy, x, y)

    def rect(self, x, y, sx, sy):
        self.path.add_rect(x, y, sx, sy)

    def rects(self, rects):
        for rect in rects:
            self.path.add_rect(rect)

    def add_path(self, other_path):
        if isinstance(other_path, CompiledPath):
            self.path.add_path(other_path.path)

    def close_path(self):
        self.path.close()

    def is_empty(self):
        return self.path.empty()

    def get_current_point(self):
        return self.path.get_last_vertex()

    def get_bounding_box(self):
        # XXX: Returns a blend2d.Rect which is sort of useless...
        self.path.get_bounding_box()


# GraphicsContext should implement AbstractGraphicsContext
AbstractGraphicsContext.register(GraphicsContext)


def font_metrics_provider():
    """ Creates an object to be used for querying font metrics.
    """
    return GraphicsContext((1, 1))
