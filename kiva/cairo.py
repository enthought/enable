# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
""" Implementation of the core2d drawing library, using cairo for rendering

    :Author:      Bryan Cole (bryan@cole.uklinux.net)
    :Copyright:   Bryan Cole (except parts copied from basecore2d)
    :License:     BSD Style

    This is currently under development and is not yet fully functional.

"""
import copy
import warnings

import cairo
import numpy

from .arc_conversion import arc_to_tangent_points
from . import basecore2d, constants


line_join = {
    constants.JOIN_BEVEL: cairo.LINE_JOIN_BEVEL,
    constants.JOIN_MITER: cairo.LINE_JOIN_MITER,
    constants.JOIN_ROUND: cairo.LINE_JOIN_ROUND,
}

line_cap = {
    constants.CAP_BUTT: cairo.LINE_CAP_BUTT,
    constants.CAP_ROUND: cairo.LINE_CAP_ROUND,
    constants.CAP_SQUARE: cairo.LINE_CAP_SQUARE,
}

font_slant = {
    "regular": cairo.FONT_SLANT_NORMAL,
    "bold": cairo.FONT_SLANT_NORMAL,
    "italic": cairo.FONT_SLANT_ITALIC,
    "bold italic": cairo.FONT_SLANT_ITALIC,
}

font_weight = {
    "regular": cairo.FONT_WEIGHT_NORMAL,
    "bold": cairo.FONT_WEIGHT_BOLD,
    "italic": cairo.FONT_WEIGHT_NORMAL,
    "bold italic": cairo.FONT_WEIGHT_BOLD,
}

spread_methods = {
    "pad": cairo.EXTEND_PAD,
    "reflect": cairo.EXTEND_REFLECT,
    "repeat": cairo.EXTEND_REPEAT,
}

text_draw_modes = {
    "FILL": (
        constants.TEXT_FILL,
        constants.TEXT_FILL_CLIP,
        constants.TEXT_FILL_STROKE,
        constants.TEXT_FILL_STROKE_CLIP,
    ),
    "STROKE": (
        constants.TEXT_FILL_STROKE,
        constants.TEXT_FILL_STROKE_CLIP,
        constants.TEXT_STROKE,
        constants.TEXT_STROKE_CLIP,
    ),
    "CLIP": (
        constants.TEXT_CLIP,
        constants.TEXT_FILL_CLIP,
        constants.TEXT_FILL_STROKE_CLIP,
        constants.TEXT_STROKE_CLIP,
    ),
    "INVISIBLE": constants.TEXT_INVISIBLE,
}


class PixelMap(object):
    def __init__(self, surface, width, height):
        self.surface = surface
        self.width = width
        self.height = height

    def draw_to_wxwindow(self, window, x, y):
        import wx

        window_dc = getattr(window, "_dc", None)
        if window_dc is None:
            window_dc = wx.PaintDC(window)
        arr = self.convert_to_rgbarray()
        image = wx.EmptyImage(self.width, self.height)
        image.SetDataBuffer(arr.data)
        bmp = wx.BitmapFromImage(image, depth=-1)
        # Scale HiDPI images to fit window
        bmp.SetSize(window.GetSize())

        window_dc.BeginDrawing()
        window_dc.DrawBitmap(bmp, x, y)
        window_dc.EndDrawing()

    def convert_to_rgbarray(self):
        pixels = numpy.frombuffer(self.surface.get_data(), numpy.uint8)

        red = pixels[2::4]
        green = pixels[1::4]
        blue = pixels[0::4]
        return numpy.vstack((red, green, blue)).T.flatten()

    def convert_to_argbarray(self, flip=False):
        pixels = numpy.frombuffer(self.surface.get_data(), numpy.uint8)

        alpha = pixels[0::4]
        red = pixels[1::4]
        green = pixels[2::4]
        blue = pixels[3::4]
        if flip:
            return (
                numpy.vstack((alpha, red, green, blue)).T
                .reshape((self.height, self.width, 4))[::-1, ...]
                .flatten()
            )
        # no flip
        return numpy.vstack((alpha, red, green, blue)).T.flatten()


class GraphicsState(object):
    """ Holds information used by a graphics context when drawing.

        The Cairo state stores the following:

        * Operator (the blend mode)
        * Tolerance
        * Antialias (bool)
        * stroke style (line width, cap, join, mitre-limit, dash-style)
        * fill rule
        * font face
        * scaled font
        * font matrix (includes font size)
        * font options (antialias, subpixel order, hint style, hint metrics)
        * clip region
        * target surface and previous target surface
        * CTM, CTM-inverse, source CTM

        The Quartz2D state (which kiva follows AFAIK) includes:

        * CTM
        * stroke style (line width, cap, join, mitre, dash)
        * clip region
        * tolerance (accuracy)
        * anti-alias
        * fill- and stroke- colors
        * fill- and stroke- Color Space (RGB, HSV, CMYK etc.)
        * Rendering intent (something to do with Color Spaces)
        * alpha value
        * blend mode
        * text font
        * text font size
        * text drawing mode (stroked, filled, clipped and combinations of
          these)
        * text character spacing (extra space between glyphs)

        *: items in the Quartz2D state that Cairo doesn't support directly.

        basecore2d GraphicsState includes:

        * ctm
        * line_color
        * line_width
        * line_join
        * line_cap
        * line_dash
        * fill_color
        * alpha
        * font
        * text_matrix
        * clipping_path
        * current_point
        * should_antialias
        * miter_limit
        * flatness
        * character_spacing
        * text_drawing_mode
        * rendering_intent (not yet implemented)

        *: discrepancies compared to Quartz2D
    """

    def __init__(self):
        self.fill_color = [1, 1, 1]
        self.stroke_color = [1, 1, 1]
        self.alpha = 1.0
        self.text_drawing_mode = constants.TEXT_FILL
        self.has_gradient = False

        # not implemented yet...
        self.text_character_spacing = None
        self.fill_colorspace = None
        self.stroke_colorspace = None
        self.rendering_intent = None

    def copy(self):
        return copy.deepcopy(self)


class GraphicsContext(basecore2d.GraphicsContextBase):
    def __init__(self, size, *args, **kw):
        super(GraphicsContext, self).__init__(size, *args, **kw)
        w, h = size

        self.surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, w, h)
        self.surface.set_device_offset(0, h)

        if "context" in kw:
            ctx = kw.pop("context")
        else:
            ctx = cairo.Context(self.surface)
            ctx.set_source_rgb(1, 1, 1)
            ctx.scale(1, -1)

        # For HiDPI support
        base_scale = kw.pop("base_pixel_scale", 1)
        ctx.scale(base_scale, base_scale)

        self._ctx = ctx
        self.state = GraphicsState()
        self.state_stack = []

        # the text-matrix includes the text position
        self.text_matrix = cairo.Matrix(
            1, 0, 0, -1, 0, 0
        )  # not part of the graphics state

        self.pixel_map = PixelMap(self.surface, w, h)

    def clear(self, color=(1, 1, 1)):
        self.save_state()
        if len(color) == 4:
            self._ctx.set_source_rgba(*color)
        else:
            self._ctx.set_source_rgb(*color)
        self.rect(0, 0, self.width(), self.height())
        self.draw_path(constants.FILL)
        self.restore_state()

    def height(self):
        return self._ctx.get_target().get_height()

    def width(self):
        return self._ctx.get_target().get_width()

    def scale_ctm(self, sx, sy):
        """ Sets the coordinate system scale to the given values, (sx,sy).

            Parameters
            ----------
            sx : float
                The new scale factor for the x axis
            sy : float
                The new scale factor for the y axis
        """
        self._ctx.scale(sx, sy)

    def translate_ctm(self, tx, ty):
        """ Translates the coordinate system by the value given by (tx,ty)

            Parameters
            ----------
            tx : float
                The distance to move in the x direction
            ty : float
                The distance to move in the y direction
        """
        self._ctx.translate(tx, ty)

    def rotate_ctm(self, angle):
        """ Rotates the coordinate space for drawing by the given angle.

            Parameters
            ----------
            angle : float
                the angle, in radians, to rotate the coordinate system
        """
        self._ctx.rotate(angle)

    def concat_ctm(self, transform):
        """ Concatenates the transform to current coordinate transform matrix.

            Parameters
            ----------
            transform : affine_matrix
                the transform matrix to concatenate with
                the current coordinate matrix.
        """
        try:
            # assume transform is a cairo.Matrix object
            self._ctx.transform(transform)
        except TypeError:
            # now assume transform is a list of matrix elements (floats)
            self._ctx.transform(cairo.Matrix(*transform))

    def get_ctm(self):
        """ Returns the current coordinate transform matrix
            as a list of matrix elements
        """
        return list(self._ctx.get_matrix())

    # ----------------------------------------------------------------
    # Save/Restore graphics state.
    # ----------------------------------------------------------------

    def save_state(self):
        """ Saves the current graphic's context state.

            Always pair this with a `restore_state()`.
        """
        self._ctx.save()
        self.state_stack.append(self.state)
        self.state = self.state.copy()

    def restore_state(self):
        """ Restores the previous graphics state.
        """
        self._ctx.restore()
        self.state = self.state_stack.pop()

    # ----------------------------------------------------------------
    # Manipulate graphics state attributes.
    # ----------------------------------------------------------------

    def set_antialias(self, value):
        """ Sets/Unsets anti-aliasing for bitmap graphics context.

            Ignored on most platforms.
        """
        if bool(value):
            val = cairo.ANTIALIAS_DEFAULT
        else:
            val = cairo.ANTIALIAS_NONE
        self._ctx.set_antialias(val)

    def set_line_width(self, width):
        """ Sets the line width for drawing

            Parameters
            ----------
            width : float
                The new width for lines in user space units.
        """
        self._ctx.set_line_width(width)

    def set_line_join(self, style):
        """ Sets the style for joining lines in a drawing.

            Parameters
            ----------
            style : join_style
                The line joining style.  The available
                styles are JOIN_ROUND, JOIN_BEVEL, JOIN_MITER.
        """
        try:
            self._ctx.set_line_join(line_join[style])
        except KeyError:
            raise ValueError("Invalid line-join style")

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
                (XXX is this the correct default?)
        """
        self._ctx.set_miter_limit(limit)

    def set_line_cap(self, style):
        """ Specifies the style of endings to put on line ends.

            Parameters
            ----------
            style : cap_style
                The line cap style to use. Available styles
                are CAP_ROUND, CAP_BUTT, CAP_SQUARE.
        """
        try:
            self._ctx.set_line_cap(line_cap[style])
        except KeyError:
            raise ValueError("Invalid line cap style")

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
        if pattern is not None:
            pattern = list(pattern)
            self._ctx.set_dash(pattern, phase)

    def set_flatness(self, flatness):
        """ Not implemented

            It is device dependent and therefore not recommended by
            the PDF documentation.

            flatness determines how accurately lines are rendered.  Setting it
            to values less than one will result in more accurate drawings, but
            they take longer.  It defaults to None
        """
        self._ctx.set_tolerance(flatness)

    # ----------------------------------------------------------------
    # Sending drawing data to a device
    # ----------------------------------------------------------------

    def flush(self):
        """ Sends all drawing data to the destination device.

            Currently this is a NOP for wxPython.
        """
        s = self._ctx.get_target()
        s.flush()

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

    def radial_gradient(self, cx, cy, r, fx, fy, stops, spreadMethod="pad",
                        units="userSpaceOnUse", transforms=None):
        """ Set a radial gradient as the fill color.
        """
        # TODO: handle transforms

        if units == "objectBoundingBox":
            # transform from relative coordinates
            path_rect = self._ctx.path_extents()
            width = path_rect[2] - path_rect[0]
            height = path_rect[3] - path_rect[1]
            r = r * width
            cx = path_rect[0] + cx * width
            fx = path_rect[0] + fx * width
            cy = path_rect[1] + cy * height
            fy = path_rect[1] + fy * height

        gradient = cairo.RadialGradient(fx, fy, 0.0, cx, cy, r)

        gradient.set_extend(
            spread_methods.get(spreadMethod, cairo.EXTEND_NONE)
        )

        for stop in stops:
            # FIXME: the stops are possibly being generated wrong if the offset
            # is specified
            if stop.size == 10:
                start = tuple(stop[0:5])
                end = tuple(stop[5:10])
                gradient.add_color_stop_rgba(*start)
                gradient.add_color_stop_rgba(*end)
            else:
                start = tuple(stop[0:5])
                gradient.add_color_stop_rgba(*start)

        self.state.has_gradient = True
        self._ctx.set_source(gradient)

    def linear_gradient(self, x1, y1, x2, y2, stops, spreadMethod="pad",
                        units="userSpaceOnUse", transforms=None):
        """ Set a linear gradient as the fill color.
        """
        # TODO: handle transforms

        if units == "objectBoundingBox":
            # transform from relative coordinates
            path_rect = self._ctx.path_extents()
            width = path_rect[2] - path_rect[0]
            height = path_rect[3] - path_rect[1]
            x1 = path_rect[0] + x1 * width
            x2 = path_rect[0] + x2 * width
            y1 = path_rect[1] + y1 * height
            y2 = path_rect[1] + y2 * height

        gradient = cairo.LinearGradient(x1, y1, x2, y2)

        gradient.set_extend(
            spread_methods.get(spreadMethod, cairo.EXTEND_NONE)
        )

        for stop in stops:
            # FIXME: the stops are possibly being generated wrong if the offset
            # is specified
            if stop.size == 10:
                start = tuple(stop[0:5])
                end = tuple(stop[5:10])
                gradient.add_color_stop_rgba(*start)
                gradient.add_color_stop_rgba(*end)
            else:
                start = tuple(stop[0:5])
                gradient.add_color_stop_rgba(*start)

        self.state.has_gradient = True
        self._ctx.set_source(gradient)

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
        self._ctx.new_path()

    def move_to(self, x, y):
        """ Starts a new drawing subpath and place the current point at (x,y).

            Notes:
                Not sure how to treat state.current_point.  Should it be the
                value of the point before or after the matrix transformation?
                It looks like before in the PDF specs.
        """
        self._ctx.move_to(x, y)

    def line_to(self, x, y):
        """ Adds a line from the current point to the given point (x,y).

            The current point is moved to (x,y).

            What should happen if move_to hasn't been called? Should it always
            begin at 0,0 or raise an error?

            Notes:
                See note in move_to about the current_point.
        """
        self._ctx.line_to(x, y)

    def lines(self, points):
        """ Adds a series of lines as a new subpath.

            Parameters
            ----------

            points
                an Nx2 array of x,y pairs

            The current_point is moved to the last point in 'points'
        """
        self._ctx.new_sub_path()
        for point in points:
            self._ctx.line_to(*point)

    def line_set(self, starts, ends):
        """ Adds a set of disjoint lines as a new subpath.

            Parameters
            ----------
            starts
                an Nx2 array of x,y pairs
            ends
                an Nx2 array of x,y pairs

            Starts and ends should have the same length.
            The current point is moved to the last point in 'ends'.

            N.B. Cairo cannot make disjointed lines as a single subpath,
            thus each line forms it's own subpath
        """
        for start, end in zip(starts, ends):
            self._ctx.move_to(*start)
            self._ctx.line_to(*end)

    def rect(self, x, y, sx, sy):
        """ Adds a rectangle as a new subpath.
        """
        self._ctx.rectangle(x, y, sx, sy)

    def close_path(self, tag=None):
        """ Closes the path of the current subpath.

            Currently starts a new subpath -- is this what we want?
            ... Cairo starts a new subpath automatically.
        """
        self._ctx.close_path()

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
        self._ctx.curve_to(x_ctrl1, y_ctrl1, x_ctrl2, y_ctrl2, x_to, y_to)

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
        if cw:  # not sure if I've got this the right way round
            self._ctx.arc_negative(x, y, radius, start_angle, end_angle)
        else:
            self._ctx.arc(x, y, radius, start_angle, end_angle)

    def arc_to(self, x1, y1, x2, y2, radius):
        """ Draw an arc between the line segments from the current point
            to (x1,y1) and from (x1,y1) to (x2,y2). Straight lines are also
            added from the current point to the start of the curve and from the
            end of the curve to (x2,y2).
        """
        current_point = self.get_path_current_point()

        # Get the endpoints on the curve where it touches the line segments
        t1, t2 = arc_to_tangent_points(
            current_point, (x1, y1), (x2, y2), radius
        )

        # draw!
        self._ctx.line_to(*t1)
        self._ctx.curve_to(x1, y1, x1, y1, *t2)
        self._ctx.line_to(x2, y2)

    # ----------------------------------------------------------------
    # Getting infomration on paths
    # ----------------------------------------------------------------

    def is_path_empty(self):
        """ Tests to see whether the current drawing path is empty

        What does 'empty' mean???
        """
        p = self._ctx.copy_path()
        return any(a[0] for a in p)

    def get_path_current_point(self):
        """ Returns the current point from the graphics context.

            Note:
                Currently the current_point is only affected by move_to,
                line_to, and lines.  It should also be affected by text
                operations.  I'm not sure how rect and rects and friends
                should affect it -- will find out on Mac.
        """
        return self._ctx.get_current_point()

    def get_path_bounding_box(self):
        """
        cairo.Context.path_extents not yet implemented on my cairo version.
        It's in new ones though.

        What should this method return?
        """
        if self.is_path_empty():
            return [[0, 0], [0, 0]]
        p = [a[1] for a in self._ctx.copy_path()]
        p = numpy.array(p)
        return [p.min(axis=1), p.max(axis=1)]

    def add_path(self, path):
        """Draw a compiled path into this gc.
        In this case, a compiled path is a Cairo.Path"""
        if isinstance(path, CompiledPath):
            self.begin_path()
            for op_name, op_args in path.state:
                op = getattr(self, op_name)
                op(*op_args)
            self.close_path()

    # ----------------------------------------------------------------
    # Clipping path manipulation
    # ----------------------------------------------------------------

    def clip(self):
        """
        Should this use clip or clip_preserve
        """
        fr = self._ctx.get_fill_rule()
        self._ctx.set_fill_rule(cairo.FILL_RULE_WINDING)
        self._ctx.clip()
        self._ctx.set_fill_rule(fr)

    def even_odd_clip(self):
        """
        """
        fr = self._ctx.get_fill_rule()
        self._ctx.set_fill_rule(cairo.FILL_RULE_EVEN_ODD)
        self._ctx.clip()
        self._ctx.set_fill_rule(fr)

    def clip_to_rect(self, x, y, width, height):
        """
            Sets the clipping path to the intersection of the current clipping
            path with the area defined by the specified rectangle
        """
        ctx = self._ctx
        # get the current path
        p = ctx.copy_path()
        ctx.new_path()
        ctx.rectangle(x, y, width, height)
        ctx.clip()
        ctx.append_path(p)

    def clear_clip_path(self):
        self._ctx.reset_clip()

    # ----------------------------------------------------------------
    # Color space manipulation
    #
    # I'm not sure we'll mess with these at all.  They seem to
    # be for setting the color system.  Hard coding to RGB or
    # RGBA for now sounds like a reasonable solution.
    # ----------------------------------------------------------------

    # def set_fill_color_space(self):
    #    """
    #    """
    #    pass

    # def set_stroke_color_space(self):
    #    """
    #    """
    #    pass

    # def set_rendering_intent(self):
    #    """
    #    """
    #    pass

    # ----------------------------------------------------------------
    # Color manipulation
    # ----------------------------------------------------------------

    def _set_source_color(self, color):
        if len(color) == 3:
            self._ctx.set_source_rgb(*color)
        else:
            self._ctx.set_source_rgba(*color)
        # gradients or other source patterns are blown away by set_source_rgb*
        self.state.has_gradient = False

    def set_fill_color(self, color):
        """ set_fill_color takes a sequences of rgb or rgba values
        between 0.0 and 1.0
        """
        self.state.fill_color = color

    def set_stroke_color(self, color):
        """ set_stroke_color takes a sequences of rgb or rgba values
        between 0.0 and 1.0
        """
        self.state.stroke_color = color

    def set_alpha(self, alpha):
        """
        """
        self.state.alpha = alpha

    # ----------------------------------------------------------------
    # Drawing Images
    # ----------------------------------------------------------------

    def draw_image(self, img, rect=None):
        """
        `img` is either a N*M*3 or N*M*4 numpy array, or a PIL Image
        `rect` is a tuple (x, y, w, h)
        """
        from PIL import Image

        if isinstance(img, numpy.ndarray):
            # Numeric array
            if img.shape[2] == 3:
                format = cairo.FORMAT_RGB24
            elif img.shape[2] == 4:
                format = cairo.FORMAT_ARGB32
            img_width, img_height = img.shape[:2]
            img_surface = cairo.ImageSurface.create_for_data(
                img.astype(numpy.uint8), format, img_width, img_height
            )
        elif isinstance(img, Image.Image):
            converted_img = img.convert("RGBA")
            img_width, img_height = img.width, img.height
            img_surface = cairo.ImageSurface.create_for_data(
                converted_img.flatten(),
                cairo.FORMAT_RGB24,
                img_width,
                img_height,
            )
        elif hasattr(img, "bmp_array"):
            # a kiva agg context
            if hasattr(img, "convert_pixel_format"):
                img = img.convert_pixel_format("rgba32", inplace=0)
            converted_img = Image.fromarray(img.bmp_array)
            flipped_array = numpy.flipud(numpy.array(converted_img))
            img_width, img_height = img.width, img.height
            img_surface = cairo.ImageSurface.create_for_data(
                flipped_array.flatten(),
                cairo.FORMAT_RGB24,
                img.width,
                img.height,
            )
        elif isinstance(img, GraphicsContext):
            # Another cairo kiva context
            img_width, img_height = img.pixel_map.width, img.pixel_map.height
            img_surface = cairo.ImageSurface.create_for_data(
                img.pixel_map.convert_to_argbarray(flip=True),
                cairo.FORMAT_ARGB32,
                img_width,
                img_height,
            )
        else:
            msg = "Cannot render image of type '%r' into cairo context."
            warnings.warn(msg % type(img))
            return

        ctx = self._ctx
        img_pattern = cairo.SurfacePattern(img_surface)
        if rect:
            x, y, sx, sy = rect
            if sx != img_width or sy != img_height:
                scaler = cairo.Matrix()
                scaler.scale(img_width / float(sx), img_height / float(sy))
                img_pattern.set_matrix(scaler)
                img_pattern.set_filter(cairo.FILTER_BEST)
            ctx.set_source(img_pattern)
            # p = ctx.copy_path() #need to save the path
            ctx.new_path()
            ctx.rectangle(x, y, sx, sy)
            ctx.fill()
        else:
            ctx.set_source(img_pattern)
            ctx.paint()

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
        # self.state.font = freetype.FontInfo(face_name,size,style,encoding)
        self._ctx.select_font_face(
            face_name, font_slant[style], font_weight[style]
        )
        self._ctx.set_font_size(size)

    def set_font(self, font):
        """ Set the font for the current graphics context.

            A device-specific font object. In this case, a cairo FontFace
            object. It's not clear how this can be used right now.
        """
        if font.weight in (constants.BOLD, constants.BOLD_ITALIC):
            weight = cairo.FONT_WEIGHT_BOLD
        else:
            weight = cairo.FONT_WEIGHT_NORMAL

        if font.style in (constants.ITALIC, constants.BOLD_ITALIC):
            style = cairo.FONT_SLANT_ITALIC
        else:
            style = cairo.FONT_SLANT_NORMAL

        face_name = font.face_name

        ctx = self._ctx
        ctx.select_font_face(face_name, style, weight)
        ctx.set_font_size(font.size)
        # facename = font.face_name
        # slant = font.style

        # self._ctx.set_font_face(font)

    def set_font_size(self, size):
        """ Sets the size of the font.

            The size is specified in user space coordinates.
        """
        self._ctx.set_font_size(size)

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
            2.  Not implemented in wxPython, or cairo (for the time being)
        """
        self.state.character_spacing = spacing

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
            constants.TEXT_FILL, constants.TEXT_STROKE,
            constants.TEXT_FILL_STROKE, constants.TEXT_INVISIBLE,
            constants.TEXT_FILL_CLIP, constants.TEXT_STROKE_CLIP,
            constants.TEXT_FILL_STROKE_CLIP, constants.TEXT_CLIP,
            constants.TEXT_OUTLINE,
        )
        if mode not in text_modes:
            msg = (
                "Invalid text drawing mode.  See documentation for valid modes"
            )
            raise ValueError(msg)
        self.state.text_drawing_mode = mode

    def set_text_position(self, x, y):
        """
        """
        m = list(self.text_matrix)
        m[4:6] = x, y
        self.text_matrix = cairo.Matrix(*m)

    def get_text_position(self):
        """
        """
        return tuple(self.text_matrix)[4:6]

    def set_text_matrix(self, ttm):
        """
        """
        if isinstance(ttm, cairo.Matrix):
            m = ttm
        else:
            m = cairo.Matrix(ttm)
        self.text_matrix = m

    def get_text_matrix(self):
        """
        """
        return copy.copy(self.text_matrix)

    def show_text(self, text, point=(0.0, 0.0)):
        """ Draws text on the device at the current text position.
            Leaves the current point unchanged.
        """
        self.show_text_at_point(text, point[0], point[1])

    def show_glyphs(self):
        """
        """
        pass

    def show_text_at_point(self, text, x, y):
        """
        """
        ctx = self._ctx
        cur_path = ctx.copy_path()
        ctx.save()
        ctx.transform(self.text_matrix)
        ctx.transform(cairo.Matrix(1, 0, 0, 1, x, y))
        ctx.new_path()
        ctx.text_path(text)
        # need to set up text drawing mode
        # 'outline' and  'invisible' modes are not supported.
        mode = self.state.text_drawing_mode
        if mode in text_draw_modes["STROKE"]:
            self._set_source_color(self.state.stroke_color)
            ctx.stroke_preserve()
        if mode in text_draw_modes["FILL"]:
            self._set_source_color(self.state.fill_color)
            ctx.fill_preserve()
        if mode in text_draw_modes["CLIP"]:
            ctx.clip_preserve()

        ctx.restore()
        ctx.new_path()
        ctx.append_path(cur_path)

    def show_glyphs_at_point(self):
        """
        """
        pass

    # ----------------------------------------------------------------
    # Painting paths (drawing and filling contours)
    # ----------------------------------------------------------------

    def draw_path(self, mode=constants.FILL_STROKE):
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
        ctx = self._ctx
        fr = ctx.get_fill_rule()
        if mode in [constants.EOF_FILL, constants.EOF_FILL_STROKE]:
            ctx.set_fill_rule(cairo.FILL_RULE_EVEN_ODD)
        else:
            ctx.set_fill_rule(cairo.FILL_RULE_WINDING)

        if mode in [constants.FILL, constants.EOF_FILL]:
            if not self.state.has_gradient:
                self._set_source_color(self.state.fill_color)
            ctx.fill()
        elif mode == constants.STROKE:
            if not self.state.has_gradient:
                self._set_source_color(self.state.stroke_color)
            ctx.stroke()
        elif mode in [constants.FILL_STROKE, constants.EOF_FILL_STROKE]:
            if not self.state.has_gradient:
                self._set_source_color(self.state.fill_color)
            ctx.fill_preserve()
            if not self.state.has_gradient:
                self._set_source_color(self.state.stroke_color)
            ctx.stroke()

        ctx.set_fill_rule(fr)

    def stroke_rect(self):
        """
        How does this affect the current path?
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

    def get_text_extent(self, textstring):
        """
            returns the width and height of the rendered text
        """
        xb, yb, w, h, xa, ya = self._ctx.text_extents(textstring)
        return xb, yb, w, h

    def get_full_text_extent(self, textstring):
        """
            How does this differ from 'get_text_extent' ???

            This just calls get_text_extent, for the time being.
        """
        x, y, w, h = self.get_text_extent(textstring)
        ascent, descent, height, maxx, maxy = self._ctx.font_extents()
        return w, ascent + descent, -descent, height

    def render_component(self, component, container_coords=False):
        """ Renders the given component.

        Parameters
        ----------
        component : Component
            The component to be rendered.
        container_coords : Boolean
            Whether to use coordinates of the component's container

        Description
        -----------
        If *container_coords* is False, then the (0,0) coordinate of this
        graphics context corresponds to the lower-left corner of the
        component's **outer_bounds**. If *container_coords* is True, then the
        method draws the component as it appears inside its container, i.e., it
        treats (0,0) of the graphics context as the lower-left corner of the
        container's outer bounds.
        """

        x, y = component.outer_position
        w, h = component.outer_bounds
        if not container_coords:
            x = -x
            y = -y
        self.translate_ctm(x, y)
        component.draw(self, view_bounds=(0, 0, w, h))

    def save(self, filename, file_format=None):
        """ Save the GraphicsContext to a (PNG) file.
            file_format is ignored.
        """
        self.surface.flush()
        self.surface.write_to_png(filename)


class CompiledPath(object):
    def __init__(self):
        self.state = []

    def add_path(self, *args):
        self.state.append(("begin_path", args))

    def rect(self, *args):
        self.state.append(("rect", args))

    def move_to(self, *args):
        self.state.append(("move_to", args))

    def line_to(self, *args):
        self.state.append(("line_to", args))

    def close_path(self, *args):
        self.state.append(("close_path", args))

    def quad_curve_to(self, *args):
        self.state.append(("quad_curve_to", args))

    def curve_to(self, *args):
        self.state.append(("curve_to", args))

    def arc(self, *args):
        self.state.append(("arc", args))

    def total_vertices(self):
        return len(self.state) + 1

    def vertex(self, index):
        return (self.state[index - 1][1][0:2],)


def font_metrics_provider():
    return GraphicsContext((1, 1))


if __name__ == "__main__":
    from numpy import linspace
    from scipy.special import jn

    from chaco.api import ArrayPlotData, Plot, PlotGraphicsContext
    from chaco.example_support import COLOR_PALETTE

    from itertools import cycle

    DPI = 72.0
    dpi_scale = DPI / 72.0

    def create_plot():
        numpoints = 100
        low = -5
        high = 15.0
        x = linspace(low, high, numpoints)
        pd = ArrayPlotData(index=x)
        p = Plot(pd, bgcolor="lightgray", padding=50, border_visible=True)
        for t, i in zip(cycle(["line", "scatter"]), range(10)):
            pd.set_data("y" + str(i), jn(i, x))
            p.plot(
                ("index", "y" + str(i)),
                color=tuple(COLOR_PALETTE[i]),
                width=2.0 * dpi_scale,
                type=t,
            )
        p.x_grid.visible = True
        p.x_grid.line_width *= dpi_scale
        p.y_grid.visible = True
        p.y_grid.line_width *= dpi_scale
        p.legend.visible = True
        return p

    container = create_plot()
    container.outer_bounds = [800, 600]
    container.do_layout(force=True)

    def render_cairo_png():
        w, h = 800, 600
        scale = 1.0
        s = cairo.ImageSurface(
            cairo.FORMAT_ARGB32, int(w * scale), int(h * scale)
        )
        s.set_device_offset(0, h * scale)
        ctx = cairo.Context(s)
        ctx.set_source_rgb(1, 1, 1)
        ctx.paint()
        ctx.scale(1, -1)
        ctx.scale(scale, scale)
        gc = GraphicsContext((w, h), context=ctx)
        gc.render_component(container)
        s.flush()
        s.write_to_png("/tmp/kiva_cairo.png")

    def render_cairo_svg():
        w, h = 800, 600
        scale = 1.0
        s = cairo.SVGSurface("/tmp/kiva_cairo.svg", w * scale, h * scale)
        s.set_device_offset(0, h * scale)
        ctx = cairo.Context(s)
        ctx.set_source_rgb(1, 1, 1)
        ctx.paint()
        ctx.scale(1, -1)
        ctx.scale(scale, scale)
        gc = GraphicsContext((w, h), context=ctx)
        gc.render_component(container)
        s.finish()

    def render_cairo_pdf():
        w, h = 800, 600
        scale = 1.0
        s = cairo.PDFSurface("/tmp/kiva_cairo.pdf", w * scale, h * scale)
        s.set_device_offset(0, h * scale)
        ctx = cairo.Context(s)
        ctx.set_source_rgb(1, 1, 1)
        ctx.paint()
        ctx.scale(1, -1)
        ctx.scale(scale, scale)
        gc = GraphicsContext((w, h), context=ctx)
        gc.render_component(container)
        s.finish()

    def render_agg():
        gc2 = PlotGraphicsContext((800, 600), dpi=DPI)
        gc2.render_component(container)
        gc2.save("/tmp/kiva_agg.png")

    # render_agg()
    render_cairo_png()
    render_cairo_svg()
    render_cairo_pdf()
    render_agg()
