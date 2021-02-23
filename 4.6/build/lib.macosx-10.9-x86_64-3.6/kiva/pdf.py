# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!

""" PDF implementation of the core2d drawing library

    :Author:      Eric Jones, Enthought, Inc., eric@enthought.com
    :Copyright:   Space Telescope Science Institute
    :License:     BSD Style

    The PDF implementation relies heavily on the ReportLab project.
"""

# standard library imports
import copy
import warnings

from numpy import ndarray, pi

# ReportLab PDF imports
import reportlab.pdfbase.pdfmetrics
import reportlab.pdfbase._fontdata
from reportlab.pdfgen import canvas

# local, relative Kiva imports
from .arc_conversion import arc_to_tangent_points
from .basecore2d import GraphicsContextBase
from .line_state import is_dashed
from .constants import FILL, STROKE, EOF_FILL
import kiva.constants as constants
import kiva.affine as affine


cap_style = {}
cap_style[constants.CAP_ROUND] = 1
cap_style[constants.CAP_SQUARE] = 2
cap_style[constants.CAP_BUTT] = 0

join_style = {}
join_style[constants.JOIN_ROUND] = 1
join_style[constants.JOIN_BEVEL] = 2
join_style[constants.JOIN_MITER] = 0

# stroke, fill, mode
path_mode = {}
path_mode[constants.FILL_STROKE] = (1, 1, canvas.FILL_NON_ZERO)
path_mode[constants.FILL] = (0, 1, canvas.FILL_NON_ZERO)
path_mode[constants.EOF_FILL] = (0, 1, canvas.FILL_EVEN_ODD)
path_mode[constants.STROKE] = (1, 0, canvas.FILL_NON_ZERO)
path_mode[constants.EOF_FILL_STROKE] = (1, 1, canvas.FILL_EVEN_ODD)


# fixme: I believe this can be implemented but for now, it is not.
class CompiledPath(object):
    pass


class GraphicsContext(GraphicsContextBase):
    """
    Simple wrapper around a PDF graphics context.
    """

    def __init__(self, pdf_canvas, *args, **kwargs):
        from .image import GraphicsContext as GraphicsContextImage

        self.gc = pdf_canvas
        self.current_pdf_path = None
        self.current_point = (0, 0)
        self.text_xy = None, None
        # get an agg backend to assist in measuring text
        self._agg_gc = GraphicsContextImage((1, 1))
        super(GraphicsContext, self).__init__(self, *args, **kwargs)

    # ----------------------------------------------------------------
    # Coordinate Transform Matrix Manipulation
    # ----------------------------------------------------------------

    def scale_ctm(self, sx, sy):
        """
        scale_ctm(sx: float, sy: float) -> None

        Sets the coordinate system scale to the given values, (sx, sy).
        """
        self.gc.scale(sx, sy)

    def translate_ctm(self, tx, ty):
        """
        translate_ctm(tx: float, ty: float) -> None

        Translates the coordinate syetem by the given value by (tx, ty)
        """
        self.gc.translate(tx, ty)

    def rotate_ctm(self, angle):
        """
        rotate_ctm(angle: float) -> None

        Rotates the coordinate space by the given angle (in radians).
        """
        self.gc.rotate(angle * 180 / pi)

    def concat_ctm(self, transform):
        """
        concat_ctm(transform: affine_matrix)

        Concatenates the transform to current coordinate transform matrix.
        transform is an affine transformation matrix (see kiva.affine_matrix).
        """
        self.gc.transform(transform)

    def get_ctm(self):
        """ Returns the current coordinate transform matrix.

            XXX: This should really return a 3x3 matrix (or maybe an affine
                 object?) like the other API's.  Needs thought.
        """
        return affine.affine_from_values(*copy.copy(self.gc._currentMatrix))

    def set_ctm(self, transform):
        """ Set the coordinate transform matrix

        """
        # We have to do this by inverting the current state to zero it out,
        # then transform by desired transform, as Reportlab Canvas doesn't
        # provide a method to directly set the ctm.
        current = self.get_ctm()
        self.concat_ctm(affine.invert(current))
        self.concat_ctm(transform)

    # ----------------------------------------------------------------
    # Save/Restore graphics state.
    # ----------------------------------------------------------------

    def save_state(self):
        """ Saves the current graphic's context state.

            Always pair this with a `restore_state()`
        """
        self.gc.saveState()

    def restore_state(self):
        """ Restores the previous graphics state.
        """
        self.gc.restoreState()

    # ----------------------------------------------------------------
    # Manipulate graphics state attributes.
    # ----------------------------------------------------------------

    def set_should_antialias(self, value):
        """ Sets/Unsets anti-aliasing for bitmap graphics context.
        """
        msg = "antialias is not part of the PDF canvas.  Should it be?"
        raise NotImplementedError(msg)

    def set_line_width(self, width):
        """ Sets the line width for drawing

            Parameters
            ----------
            width : float
                The new width for lines in user space units.
        """
        self.gc.setLineWidth(width)

    def set_line_join(self, style):
        """ Sets style for joining lines in a drawing.

            style : join_style
                The line joining style.  The available
                styles are JOIN_ROUND, JOIN_BEVEL, JOIN_MITER.
        """
        try:
            sjoin = join_style[style]
        except KeyError:
            msg = "Invalid line join style. See documentation for valid styles"
            raise ValueError(msg)
        self.gc.setLineJoin(sjoin)

    def set_miter_limit(self, limit):
        """ Specifies limits on line lengths for mitering line joins.

            If line_join is set to miter joins, the limit specifies which
            line joins should actually be mitered.  If lines aren't mitered,
            they are joined with a bevel.  The line width is divided by
            the length of the miter.  If the result is greater than the
            limit, the bevel style is used.

            Parameters
            ----------
            limit : float
                limit for mitering joins.
        """
        self.gc.setMiterLimit(limit)

    def set_line_cap(self, style):
        """ Specifies the style of endings to put on line ends.

            Parameters
            ----------
            style : cap_style
                the line cap style to use. Available styles
                are CAP_ROUND, CAP_BUTT, CAP_SQUARE
        """
        try:
            scap = cap_style[style]
        except KeyError:
            msg = "Invalid line cap style.  See documentation for valid styles"
            raise ValueError(msg)
        self.gc.setLineCap(scap)

    def set_line_dash(self, lengths, phase=0):
        """
            Parameters
            ----------
            lengths : float array
                An array of floating point values
                specifing the lengths of on/off painting
                pattern for lines.
            phase : float
                Specifies how many units into dash pattern
                to start.  phase defaults to 0.
        """
        if is_dashed((phase, lengths)):
            lengths = list(lengths) if lengths is not None else []
            self.gc.setDash(lengths, phase)

    def set_flatness(self, flatness):
        """
            It is device dependent and therefore not recommended by
            the PDF documentation.
        """
        raise NotImplementedError("Flatness not implemented yet on PDF")

    # ----------------------------------------------------------------
    # Sending drawing data to a device
    # ----------------------------------------------------------------

    def flush(self):
        """ Sends all drawing data to the destination device.

            Currently, this is a NOP.  It used to call ReportLab's save()
            method, and maybe it still should, but flush() is likely to
            be called a lot, so this will really slow things down.  Also,
            I think save() affects the paging of a document I think.
            We'll have to look into this more.
        """
        pass

    def synchronize(self):
        """ Prepares drawing data to be updated on a destination device.

            Currently, doesn't do anything.
            Should this call ReportLab's canvas object's showPage() method.
        """
        pass

    # ----------------------------------------------------------------
    # Page Definitions
    # ----------------------------------------------------------------

    def begin_page(self):
        """ Creates a new page within the graphics context.

            Currently, this just calls ReportLab's canvas object's
            showPage() method.  Not sure about this...
        """
        self.gc.showPage()

    def end_page(self):
        """ Ends drawing in the current page of the graphics context.

            Currently, this just calls ReportLab's canvas object's
            showPage() method.  Not sure about this...
        """
        self.gc.showPage()

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
        """ Clears the current drawing path and begins a new one.
        """
        self.current_pdf_path = self.gc.beginPath()
        self.current_point = (0, 0)

    def move_to(self, x, y):
        """ Starts a new drawing subpath at place the current point at (x, y).
        """
        if self.current_pdf_path is None:
            self.begin_path()

        self.current_pdf_path.moveTo(x, y)
        self.current_point = (x, y)

    def line_to(self, x, y):
        """ Adds a line from the current point to the given point (x, y).

            The current point is moved to (x, y).
        """
        if self.current_pdf_path is None:
            self.begin_path()

        self.current_pdf_path.lineTo(x, y)
        self.current_point = (x, y)

    def lines(self, points):
        """ Adds a series of lines as a new subpath.

            Currently implemented by calling line_to a zillion times.

            Points is an Nx2 array of x, y pairs.

            current_point is moved to the last point in points
        """
        if self.current_pdf_path is None:
            self.begin_path()

        self.current_pdf_path.moveTo(points[0][0], points[0][1])
        for x, y in points[1:]:
            self.current_pdf_path.lineTo(x, y)
            self.current_point = (x, y)

    def line_set(self, starts, ends):
        if self.current_pdf_path is None:
            self.begin_path()

        for start, end in zip(starts, ends):
            self.current_pdf_path.moveTo(start[0], start[1])
            self.current_pdf_path.lineTo(end[0], end[1])
            self.current_point = (end[0], end[1])

    def rect(self, *args):
        """ Adds a rectangle as a new subpath.  Can be called in two ways:
              rect(x, y, w, h)
              rect( (x, y, w, h) )

        """
        if self.current_pdf_path is None:
            self.begin_path()

        if len(args) == 1:
            args = args[0]
        self.current_pdf_path.rect(*args)
        self.current_point = (args[0], args[1])

    def draw_rect(self, rect, mode=constants.FILL_STROKE):
        self.rect(rect)
        self.draw_path(mode)
        self.current_point = (rect[0], rect[1])

    def rects(self, rects):
        """ Adds multiple rectangles as separate subpaths to the path.

            Currently implemented by calling rect a zillion times.

        """
        if self.current_pdf_path is None:
            self.begin_path()

        for x, y, sx, sy in rects:
            self.current_pdf_path.rect(x, y, sx, sy)
            self.current_point = (x, y)

    def close_path(self):
        """ Closes the path of the current subpath.
        """
        self.current_pdf_path.close()

    def curve_to(self, cp1x, cp1y, cp2x, cp2y, x, y):
        """
        """
        if self.current_pdf_path is None:
            self.begin_path()

        self.current_pdf_path.curveTo(cp1x, cp1y, cp2x, cp2y, x, y)
        self.current_point = (x, y)

    def quad_curve_to(self, cpx, cpy, x, y):
        """
        """
        msg = "quad curve to not implemented yet on PDF"
        raise NotImplementedError(msg)

    def arc(self, x, y, radius, start_angle, end_angle, clockwise=False):
        """
        """
        if self.current_pdf_path is None:
            self.begin_path()

        self.current_pdf_path.arc(
            x - radius, y - radius,
            x + radius, y + radius,
            start_angle * 180.0 / pi,
            (end_angle - start_angle) * 180.0 / pi,
        )
        self.current_point = (x, y)

    def arc_to(self, x1, y1, x2, y2, radius):
        """
        """
        if self.current_pdf_path is None:
            self.begin_path()

        # Get the endpoints on the curve where it touches the line segments
        t1, t2 = arc_to_tangent_points(
            self.current_point, (x1, y1), (x2, y2), radius
        )

        # draw!
        self.current_pdf_path.lineTo(*t1)
        self.current_pdf_path.curveTo(x1, y1, x1, y1, *t2)
        self.current_pdf_path.lineTo(x2, y2)
        self.current_point = (x2, y2)

    # ----------------------------------------------------------------
    # Getting infomration on paths
    # ----------------------------------------------------------------

    def is_path_empty(self):
        """ Tests to see whether the current drawing path is empty
        """
        msg = "is_path_empty not implemented yet on PDF"
        raise NotImplementedError(msg)

    def get_path_current_point(self):
        """ Returns the current point from the graphics context.

            Note: This should be a tuple or array.

        """
        return self.current_point

    def get_path_bounding_box(self):
        """
            Should return a tuple or array instead of a strange object.
        """
        msg = "get_path_bounding_box not implemented yet on PDF"
        raise NotImplementedError(msg)

    # ----------------------------------------------------------------
    # Clipping path manipulation
    # ----------------------------------------------------------------

    def clip(self):
        """
        """
        self.gc._fillMode = canvas.FILL_NON_ZERO
        self.gc.clipPath(self.current_pdf_path, stroke=0, fill=0)

    def even_odd_clip(self):
        """
        """
        self.gc._fillMode = canvas.FILL_EVEN_ODD
        self.gc.clipPath(self.current_pdf_path, stroke=0, fill=1)

    def clip_to_rect(self, x, y, width, height):
        """ Clips context to the given rectangular region.

            Region should be a 4-tuple or a sequence.
        """
        clip_path = self.gc.beginPath()
        clip_path.rect(x, y, width, height)
        self.gc.clipPath(clip_path, stroke=0, fill=0)

    def clip_to_rects(self):
        """
        """
        msg = "clip_to_rects not implemented yet on PDF."
        raise NotImplementedError(msg)

    def clear_clip_path(self):
        """
        """

        return
        self.clip_to_rect(0, 0, 10000, 10000)

    # ----------------------------------------------------------------
    # Color space manipulation
    #
    # I'm not sure we'll mess with these at all.  They seem to
    # be for setting the color syetem.  Hard coding to RGB or
    # RGBA for now sounds like a reasonable solution.
    # ----------------------------------------------------------------

    def set_fill_color_space(self):
        """
        """
        msg = "set_fill_color_space not implemented on PDF yet."
        raise NotImplementedError(msg)

    def set_stroke_color_space(self):
        """
        """
        msg = "set_stroke_color_space not implemented on PDF yet."
        raise NotImplementedError(msg)

    def set_rendering_intent(self):
        """
        """
        msg = "set_rendering_intent not implemented on PDF yet."
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
        self.gc.setFillColorRGB(r, g, b, a)

    def set_stroke_color(self, color):
        """
        """
        r, g, b = color[:3]
        try:
            a = color[3]
        except IndexError:
            a = 1.0
        self.gc.setStrokeColorRGB(r, g, b, a)

    def set_alpha(self, alpha):
        """ Sets alpha globally. Note that this will not affect draw_image
            because reportlab does not currently support drawing images with
            alpha.
        """
        self.gc.setFillAlpha(alpha)
        self.gc.setStrokeAlpha(alpha)
        super(GraphicsContext, self).set_alpha(alpha)

    # ----------------------------------------------------------------
    # Drawing Images
    # ----------------------------------------------------------------

    def draw_image(self, img, rect=None):
        """
        draw_image(img_gc, rect=(x, y, w, h))

        Draws another gc into this one.  If 'rect' is not provided, then
        the image gc is drawn into this one, rooted at (0, 0) and at full
        pixel size.  If 'rect' is provided, then the image is resized
        into the (w, h) given and drawn into this GC at point (x, y).

        img_gc is either a Numeric array (WxHx3 or WxHx4) or a PIL Image.

        Requires the Python Imaging Library (PIL).
        """

        # We turn img into a PIL object, since that is what ReportLab
        # requires.  To do this, we first determine if the input image
        # GC needs to be converted to RGBA/RGB.  If so, we see if we can
        # do it nicely (using convert_pixel_format), and if not, we do
        # it brute-force using Agg.
        from reportlab.lib.utils import ImageReader
        from PIL import Image

        if isinstance(img, ndarray):
            # Conversion from numpy array
            pil_img = Image.fromarray(img)
        elif isinstance(img, Image.Image):
            pil_img = img
        elif hasattr(img, "bmp_array"):
            # An offscreen kiva agg context
            if hasattr(img, "convert_pixel_format"):
                img = img.convert_pixel_format("rgba32", inplace=0)
            pil_img = Image.fromarray(img.bmp_array)
        else:
            warnings.warn(
                "Cannot render image of type %r into PDF context." % type(img)
            )
            return

        if rect is None:
            rect = (0, 0, pil_img.width, pil_img.height)

        # Draw the actual image.
        # Wrap it in an ImageReader object, because that's what reportlab
        # actually needs.
        self.gc.drawImage(
            ImageReader(pil_img), rect[0], rect[1], rect[2], rect[3]
        )

    # ----------------------------------------------------------------
    # Drawing Text
    # ----------------------------------------------------------------

    def select_font(self, name, size, textEncoding):
        """ PDF ignores the Encoding variable.
        """
        self.gc.setFont(name, size)

    def set_font(self, font):
        """ Sets the font for the current graphics context.
        """
        # TODO: Make this actually do the right thing
        face_name = font.face_name
        if face_name == "":
            face_name = "Helvetica"
        self.gc.setFont(face_name, font.size)

    def get_font(self):
        """ Get the current font """
        raise NotImplementedError

    def set_font_size(self, size):
        """
        """
        font = self.gc._fontname
        self.gc.setFont(font, size)

    def set_character_spacing(self):
        """
        """
        pass

    def get_character_spacing(self):
        """ Get the current font """
        raise NotImplementedError

    def set_text_drawing_mode(self):
        """
        """
        pass

    def set_text_position(self, x, y):
        """
        """
        self.text_xy = x, y

    def get_text_position(self):
        """
        """
        return self.state.text_matrix[2, :2]

    def set_text_matrix(self, ttm):
        """
        """
        a, b, c, d, tx, ty = affine.affine_params(ttm)
        self.gc._textMatrix = (a, b, c, d, tx, ty)

    def get_text_matrix(self):
        """
        """
        a, b, c, d, tx, ty = self.gc._textMatrix
        return affine.affine_from_values(a, b, c, d, tx, ty)

    def show_text(self, text, x=None, y=None):
        """ Draws text on the device at current text position.

            This is also used for showing text at a particular point
            specified by x and y.

            This ignores the text matrix for now.
        """
        if x and y:
            pass
        else:
            x, y = self.text_xy
        self.gc.drawString(x, y, text)

    def show_text_at_point(self, text, x, y):
        self.show_text(text, x, y)

    def show_glyphs(self):
        """
        """
        msg = "show_glyphs not implemented on PDF yet."
        raise NotImplementedError(msg)

    def get_full_text_extent(self, textstring):
        fontname = self.gc._fontname
        fontsize = self.gc._fontsize

        ascent, descent = reportlab.pdfbase._fontdata.ascent_descent[fontname]

        # get the AGG extent (we just care about the descent)
        aw, ah, ad, al = self._agg_gc.get_full_text_extent(textstring)

        # ignore the descent returned by reportlab if AGG returned 0.0 descent
        descent = 0.0 if ad == 0.0 else descent * fontsize / 1000.0
        ascent = ascent * fontsize / 1000.0
        height = ascent + abs(descent)
        width = self.gc.stringWidth(textstring, fontname, fontsize)

        # the final return value is defined as leading. do not know
        # how to get that number so returning zero
        return width, height, descent, 0

    def get_text_extent(self, textstring):
        w, h, *_ = self.get_full_text_extent(textstring)
        return w, h

    # ----------------------------------------------------------------
    # Painting paths (drawing and filling contours)
    # ----------------------------------------------------------------

    def clear(self):
        """
        """
        warnings.warn("clear() is ignored for the pdf backend")

    def stroke_path(self):
        """
        """
        self.draw_path(mode=STROKE)

    def fill_path(self):
        """
        """
        self.draw_path(mode=FILL)

    def eof_fill_path(self):
        """
        """
        self.draw_path(mode=EOF_FILL)

    def stroke_rect(self, rect):
        """
        """
        self.begin_path()
        self.rect(rect[0], rect[1], rect[2], rect[3])
        self.stroke_path()

    def stroke_rect_with_width(self, rect, width):
        """
        """
        msg = "stroke_rect_with_width not implemented on PDF yet."
        raise NotImplementedError(msg)

    def fill_rect(self, rect):
        """
        """
        self.begin_path()
        self.rect(rect[0], rect[1], rect[2], rect[3])
        self.fill_path()

    def fill_rects(self):
        """
        """
        msg = "fill_rects not implemented on PDF yet."
        raise NotImplementedError(msg)

    def clear_rect(self, rect):
        """
        """
        msg = "clear_rect not implemented on PDF yet."
        raise NotImplementedError(msg)

    def draw_path(self, mode=constants.FILL_STROKE):
        """ Walks through all the drawing subpaths and draw each element.

            Each subpath is drawn separately.
        """
        if self.current_pdf_path is not None:
            stroke, fill, mode = path_mode[mode]
            self.gc._fillMode = mode
            self.gc.drawPath(self.current_pdf_path, stroke=stroke, fill=fill)
            # erase the current path.
            self.current_pdf_path = None

    def save(self):
        self.gc.save()
