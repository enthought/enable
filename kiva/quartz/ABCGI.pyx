# (C) Copyright 2004-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
from __future__ import print_function

include "Python.pxi"
include "CoreFoundation.pxi"
include "CoreGraphics.pxi"
include "CoreText.pxi"

cimport c_numpy

import os
import warnings

from CTFont import default_font_info

cdef extern from "math.h":
    double sqrt(double arg)
    int isnan(double arg)


cdef CFURLRef url_from_filename(char* filename) except NULL:
    cdef CFStringRef filePath
    filePath = CFStringCreateWithCString(NULL, filename,
        kCFStringEncodingUTF8)
    if filePath == NULL:
        raise RuntimeError("could not create CFStringRef")

    cdef CFURLRef cfurl
    cfurl = CFURLCreateWithFileSystemPath(NULL, filePath,
        kCFURLPOSIXPathStyle, 0)
    CFRelease(filePath)
    if cfurl == NULL:
        raise RuntimeError("could not create a CFURLRef")
    return cfurl

cdef CGRect CGRectMakeFromPython(object seq):
    return CGRectMake(seq[0], seq[1], seq[2], seq[3])

# Enumerations

class LineCap:
    butt = kCGLineCapButt
    round = kCGLineCapRound
    square = kCGLineCapSquare

class LineJoin:
    miter = kCGLineJoinMiter
    round = kCGLineJoinRound
    bevel = kCGLineJoinBevel

class PathDrawingMode:
    fill = kCGPathFill
    eof_fill = kCGPathEOFill
    stroke = kCGPathStroke
    fill_stroke = kCGPathFillStroke
    eof_fill_stroke = kCGPathEOFillStroke

class RectEdge:
    min_x_edge = CGRectMinXEdge
    min_y_edge = CGRectMinYEdge
    max_x_edge = CGRectMaxXEdge
    max_y_edge = CGRectMaxYEdge

class ColorRenderingIntent:
    default = kCGRenderingIntentDefault
    absolute_colorimetric = kCGRenderingIntentAbsoluteColorimetric
    realative_colorimetric = kCGRenderingIntentRelativeColorimetric
    perceptual = kCGRenderingIntentPerceptual
    saturation = kCGRenderingIntentSaturation

#class ColorSpaces:
#    gray = kCGColorSpaceUserGray
#    rgb = kCGColorSpaceUserRGB
#    cmyk = kCGColorSpaceUserCMYK

class FontEnum:
    index_max  = kCGFontIndexMax
    index_invalid  = kCGFontIndexInvalid
    glyph_max  = kCGGlyphMax

class TextDrawingMode:
    fill = kCGTextFill
    stroke = kCGTextStroke
    fill_stroke = kCGTextFillStroke
    invisible = kCGTextInvisible
    fill_clip = kCGTextFillClip
    stroke_clip = kCGTextStrokeClip
    fill_stroke_clip = kCGTextFillStrokeClip
    clip = kCGTextClip

class TextEncodings:
    font_specific = kCGEncodingFontSpecific
    mac_roman = kCGEncodingMacRoman

class ImageAlphaInfo:
    none = kCGImageAlphaNone
    premultiplied_last = kCGImageAlphaPremultipliedLast
    premultiplied_first = kCGImageAlphaPremultipliedFirst
    last = kCGImageAlphaLast
    first = kCGImageAlphaFirst
    none_skip_last = kCGImageAlphaNoneSkipLast
    none_skip_first = kCGImageAlphaNoneSkipFirst
    only = kCGImageAlphaOnly

class InterpolationQuality:
    default = kCGInterpolationDefault
    none = kCGInterpolationNone
    low = kCGInterpolationLow
    high = kCGInterpolationHigh

class PathElementType:
    move_to = kCGPathElementMoveToPoint,
    line_to = kCGPathElementAddLineToPoint,
    quad_curve_to = kCGPathElementAddQuadCurveToPoint,
    curve_to = kCGPathElementAddCurveToPoint,
    close_path = kCGPathElementCloseSubpath

class StringEncoding:
    mac_roman  = kCFStringEncodingMacRoman
    windows_latin1  = kCFStringEncodingWindowsLatin1
    iso_latin1  = kCFStringEncodingISOLatin1
    nextstep_latin  = kCFStringEncodingNextStepLatin
    ascii  = kCFStringEncodingASCII
    unicode  = kCFStringEncodingUnicode
    utf8  = kCFStringEncodingUTF8
    nonlossy_ascii  = kCFStringEncodingNonLossyASCII

class URLPathStyle:
    posix = kCFURLPOSIXPathStyle
    hfs = kCFURLHFSPathStyle
    windows = kCFURLWindowsPathStyle

c_numpy.import_array()
import numpy

from kiva import constants

cap_style = {}
cap_style[constants.CAP_ROUND]  = kCGLineCapRound
cap_style[constants.CAP_SQUARE] = kCGLineCapSquare
cap_style[constants.CAP_BUTT]   = kCGLineCapButt

join_style = {}
join_style[constants.JOIN_ROUND] = kCGLineJoinRound
join_style[constants.JOIN_BEVEL] = kCGLineJoinBevel
join_style[constants.JOIN_MITER] = kCGLineJoinMiter

draw_modes = {}
draw_modes[constants.FILL]            = kCGPathFill
draw_modes[constants.EOF_FILL]        = kCGPathEOFill
draw_modes[constants.STROKE]          = kCGPathStroke
draw_modes[constants.FILL_STROKE]     = kCGPathFillStroke
draw_modes[constants.EOF_FILL_STROKE] = kCGPathEOFillStroke

text_modes = {}
text_modes[constants.TEXT_FILL]             = kCGTextFill
text_modes[constants.TEXT_STROKE]           = kCGTextStroke
text_modes[constants.TEXT_FILL_STROKE]      = kCGTextFillStroke
text_modes[constants.TEXT_INVISIBLE]        = kCGTextInvisible
text_modes[constants.TEXT_FILL_CLIP]        = kCGTextFillClip
text_modes[constants.TEXT_STROKE_CLIP]      = kCGTextStrokeClip
text_modes[constants.TEXT_FILL_STROKE_CLIP] = kCGTextFillStrokeClip
text_modes[constants.TEXT_CLIP]             = kCGTextClip
# this last one doesn't exist in Quartz
text_modes[constants.TEXT_OUTLINE]          = kCGTextStroke

cdef class CGContext
cdef class CGContextInABox(CGContext)
cdef class CGImage
cdef class CGPDFDocument
cdef class Rect
cdef class CGLayerContext(CGContextInABox)
cdef class CGBitmapContext(CGContext)
cdef class CGPDFContext(CGContext)
cdef class CGImageMask(CGImage)
cdef class CGAffine
cdef class CGMutablePath
cdef class Shading
cdef class ShadingFunction

cdef class CGContext:
    cdef CGContextRef context
    cdef float base_scale
    cdef long can_release
    cdef object current_font
    cdef object current_style
    cdef CGAffineTransform text_matrix
    cdef object font_cache
    cdef object fill_color
    cdef object stroke_color

    def __cinit__(self, *args, **kwds):
        self.context = NULL
        self.can_release = 0
        self.text_matrix = CGAffineTransformMake(1.0, 0.0, 0.0, 1.0, 0.0, 0.0)

    def __init__(self, size_t context, long can_release=0, base_pixel_scale=1.0):
        self.context = <CGContextRef>context

        self.base_scale = base_pixel_scale
        self.can_release = can_release
        self.fill_color = (0.0, 0.0, 0.0, 1.0)
        self.stroke_color = (0.0, 0.0, 0.0, 1.0)

        self._setup_color_space()
        self._setup_fonts()

    def _setup_color_space(self):
        # setup an RGB color space
        cdef CGColorSpaceRef space

        space = CGColorSpaceCreateDeviceRGB()
        CGContextSetFillColorSpace(self.context, space)
        CGContextSetStrokeColorSpace(self.context, space)
        CGColorSpaceRelease(space)

    def _setup_fonts(self):
        self.current_font = None
        self.current_style = None
        self.font_cache = {}
        self.select_font("Helvetica", 12)
        CGContextSetShouldSmoothFonts(self.context, 1)
        CGContextSetShouldAntialias(self.context, 1)
        CGContextSetTextMatrix(self.context, CGAffineTransformIdentity);
    #----------------------------------------------------------------
    # Coordinate Transform Matrix Manipulation
    #----------------------------------------------------------------

    def scale_ctm(self, float sx, float sy):
        """ Set the coordinate system scale to the given values, (sx,sy).

            sx:float -- The new scale factor for the x axis
            sy:float -- The new scale factor for the y axis
        """
        CGContextScaleCTM(self.context, sx, sy)

    def translate_ctm(self, float tx, float ty):
        """ Translate the coordinate system by the given value by (tx,ty)

            tx:float --  The distance to move in the x direction
            ty:float --   The distance to move in the y direction
        """
        CGContextTranslateCTM(self.context, tx, ty)

    def rotate_ctm(self, float angle):
        """ Rotates the coordinate space for drawing by the given angle.

            angle:float -- the angle, in radians, to rotate the coordinate
                           system
        """
        CGContextRotateCTM(self.context, angle)

    def concat_ctm(self, object transform):
        """ Concatenate the transform to current coordinate transform matrix.

            transform:affine_matrix -- the transform matrix to concatenate with
                                       the current coordinate matrix.
        """
        cdef float a,b,c,d,tx,ty
        a,b,c,d,tx,ty = transform

        cdef CGAffineTransform atransform
        atransform = CGAffineTransformMake(a,b,c,d,tx,ty)

        CGContextConcatCTM(self.context, atransform)

    def get_ctm(self):
        """ Return the current coordinate transform matrix.
        """
        cdef CGAffineTransform t
        t = CGContextGetCTM(self.context)
        return (t.a, t.b, t.c, t.d, t.tx, t.ty)

    def get_ctm_scale(self):
        """ Returns the average scaling factor of the transform matrix.

        This isn't really part of the GC interface, but it is a convenience
        method to make up for us not having full AffineMatrix support in the
        Mac backend.
        """
        cdef CGAffineTransform t
        t = CGContextGetCTM(self.context)
        x = sqrt(2.0) / 2.0 * (t.a + t.b)
        y = sqrt(2.0) / 2.0 * (t.c + t.d)
        return sqrt(x*x + y*y)



    #----------------------------------------------------------------
    # Save/Restore graphics state.
    #----------------------------------------------------------------

    def save_state(self):
        """ Save the current graphic's context state.

            This should always be paired with a restore_state
        """
        CGContextSaveGState(self.context)

    def restore_state(self):
        """ Restore the previous graphics state.
        """
        CGContextRestoreGState(self.context)

    #----------------------------------------------------------------
    # context manager interface
    #----------------------------------------------------------------

    def __enter__(self):
        self.save_state()

    def __exit__(self, object type, object value, object traceback):
        self.restore_state()

    #----------------------------------------------------------------
    # Manipulate graphics state attributes.
    #----------------------------------------------------------------

    def set_antialias(self, bool value):
        """ Set/Unset antialiasing for bitmap graphics context.
        """
        CGContextSetShouldAntialias(self.context, value)

    def set_line_width(self, float width):
        """ Set the line width for drawing

            width:float -- The new width for lines in user space units.
        """
        CGContextSetLineWidth(self.context, width)

    def set_line_join(self, object style):
        """ Set style for joining lines in a drawing.

            style:join_style -- The line joining style.  The available
                                styles are JOIN_ROUND, JOIN_BEVEL, JOIN_MITER.
        """
        try:
            sjoin = join_style[style]
        except KeyError:
            msg = "Invalid line join style.  See documentation for valid styles"
            raise ValueError(msg)
        CGContextSetLineJoin(self.context, sjoin)

    def set_miter_limit(self, float limit):
        """ Specifies limits on line lengths for mitering line joins.

            If line_join is set to miter joins, the limit specifies which
            line joins should actually be mitered.  If lines aren't mitered,
            they are joined with a bevel.  The line width is divided by
            the length of the miter.  If the result is greater than the
            limit, the bevel style is used.

            limit:float -- limit for mitering joins.
        """
        CGContextSetMiterLimit(self.context, limit)

    def set_line_cap(self, object style):
        """ Specify the style of endings to put on line ends.

            style:cap_style -- the line cap style to use. Available styles
                               are CAP_ROUND,CAP_BUTT,CAP_SQUARE
        """
        try:
            scap = cap_style[style]
        except KeyError:
            msg = "Invalid line cap style.  See documentation for valid styles"
            raise ValueError(msg)
        CGContextSetLineCap(self.context, scap)

    def set_line_dash(self, object lengths, float phase=0.0):
        """
            lengths:float array -- An array of floating point values
                                   specifing the lengths of on/off painting
                                   pattern for lines.
            phase:float -- Specifies how many units into dash pattern
                           to start.  phase defaults to 0.
        """
        cdef int n
        cdef int i
        cdef CGFloat *flengths

        if lengths is None:
            # No dash; solid line.
            CGContextSetLineDash(self.context, 0.0, NULL, 0)
            return
        else:
            n = len(lengths)
            flengths = <CGFloat*>PyMem_Malloc(n*sizeof(CGFloat))
            if flengths == NULL:
                raise MemoryError("could not allocate %s floats" % n)
            for i from 0 <= i < n:
                flengths[i] = lengths[i]
            CGContextSetLineDash(self.context, phase, flengths, n)
            PyMem_Free(flengths)

    def set_flatness(self, float flatness):
        """
            It is device dependent and therefore not recommended by
            the PDF documentation.
        """
        CGContextSetFlatness(self.context, flatness)

    #----------------------------------------------------------------
    # Sending drawing data to a device
    #----------------------------------------------------------------

    def flush(self):
        """ Send all drawing data to the destination device.
        """
        CGContextFlush(self.context)

    def synchronize(self):
        """ Prepares drawing data to be updated on a destination device.
        """
        CGContextSynchronize(self.context)

    #----------------------------------------------------------------
    # Page Definitions
    #----------------------------------------------------------------

    def begin_page(self, media_box=None):
        """ Create a new page within the graphics context.
        """
        cdef CGRect mbox
        cdef CGRect* mbox_ptr
        if media_box is not None:
            mbox = CGRectMakeFromPython(media_box)
            mbox_ptr = &mbox
        else:
            mbox_ptr = NULL

        CGContextBeginPage(self.context, mbox_ptr)

    def end_page(self):
        """ End drawing in the current page of the graphics context.
        """
        CGContextEndPage(self.context)

    #----------------------------------------------------------------
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
    #----------------------------------------------------------------

    def begin_path(self):
        """ Clear the current drawing path and begin a new one.
        """
        CGContextBeginPath(self.context)

    def move_to(self, float x, float y):
        """ Start a new drawing subpath at place the current point at (x,y).
        """
        CGContextMoveToPoint(self.context, x,y)

    def line_to(self, float x, float y):
        """ Add a line from the current point to the given point (x,y).

            The current point is moved to (x,y).
        """
        CGContextAddLineToPoint(self.context, x,y)

    def lines(self, object points):
        """ Add a series of lines as a new subpath.

            Points is an Nx2 array of x,y pairs.

            current_point is moved to the last point in points
        """

        cdef int n
        cdef int i
        cdef c_numpy.ndarray apoints
        cdef float x, y

        n = len(points)

        # Shortcut for the 0 and 1 point case
        if n < 2:
            return

        apoints = <c_numpy.ndarray>(numpy.asarray(points, dtype=numpy.float32))

        if apoints.nd != 2 or apoints.dimensions[1] != 2:
            msg = "must pass array of 2-D points"
            raise ValueError(msg)

        x = (<float*>c_numpy.PyArray_GETPTR2(apoints, 0, 0))[0]
        y = (<float*>c_numpy.PyArray_GETPTR2(apoints, 0, 1))[0]
        CGContextMoveToPoint(self.context, x, y)
        for i from 1 <= i < n:
            x = (<float*>c_numpy.PyArray_GETPTR2(apoints, i, 0))[0]
            y = (<float*>c_numpy.PyArray_GETPTR2(apoints, i, 1))[0]
            CGContextAddLineToPoint(self.context, x, y)

    def line_set(self, object starts, object ends):
        """ Adds a series of disconnected line segments as a new subpath.

            starts and ends are Nx2 arrays of (x,y) pairs indicating the
            starting and ending points of each line segment.

            current_point is moved to the last point in ends
        """
        cdef int n
        n = len(starts)
        if len(ends) < n:
            n = len(ends)

        cdef int i
        for i from 0 <= i < n:
            CGContextMoveToPoint(self.context, starts[i][0], starts[i][1])
            CGContextAddLineToPoint(self.context, ends[i][0], ends[i][1])

    def rect(self, float x, float y, float sx, float sy):
        """ Add a rectangle as a new subpath.
        """
        CGContextAddRect(self.context, CGRectMake(x,y,sx,sy))

    def rects(self, object rects):
        """ Add multiple rectangles as separate subpaths to the path.
        """
        cdef int n
        n = len(rects)
        cdef int i
        for i from 0 <= i < n:
            CGContextAddRect(self.context, CGRectMakeFromPython(rects[i]))

    def close_path(self):
        """ Close the path of the current subpath.
        """
        if not CGContextIsPathEmpty(self.context):
            CGContextClosePath(self.context)

    def curve_to(self, float cp1x, float cp1y, float cp2x, float cp2y,
        float x, float y):
        """
        """
        CGContextAddCurveToPoint(self.context, cp1x, cp1y, cp2x, cp2y, x, y )

    def quad_curve_to(self, float cpx, float cpy, float x, float y):
        """
        """
        CGContextAddQuadCurveToPoint(self.context, cpx, cpy, x, y)

    def arc(self, float x, float y, float radius, float start_angle,
        float end_angle, bool clockwise=False):
        """
        """
        CGContextAddArc(self.context, x, y, radius, start_angle, end_angle,
                           clockwise)

    def arc_to(self, float x1, float y1, float x2, float y2, float radius):
        """
        """
        CGContextAddArcToPoint(self.context, x1, y1, x2, y2, radius)

    def add_path(self, CGMutablePath path not None):
        """
        """
        CGContextAddPath(self.context, path.path)

    #----------------------------------------------------------------
    # Getting information on paths
    #----------------------------------------------------------------

    def is_path_empty(self):
        """ Test to see if the current drawing path is empty
        """
        return CGContextIsPathEmpty(self.context)

    def get_path_current_point(self):
        """ Return the current point from the graphics context.

            Note: This should be a tuple or array.

        """
        cdef CGPoint result
        result = CGContextGetPathCurrentPoint(self.context)
        return result.x, result.y

    def get_path_bounding_box(self):
        """
            should return a tuple or array instead of a strange object.
        """
        cdef CGRect result
        result = CGContextGetPathBoundingBox(self.context)
        return (result.origin.x, result.origin.y,
                result.size.width, result.size.height)

    #----------------------------------------------------------------
    # Clipping path manipulation
    #----------------------------------------------------------------

    def clip(self):
        """
        """
        if not CGContextIsPathEmpty(self.context):
            CGContextClip(self.context)

    def even_odd_clip(self):
        """
        """
        CGContextEOClip(self.context)

    def clip_to_rect(self, float x, float y, float width, float height):
        """ Clip context to the given rectangular region.
        """
        CGContextClipToRect(self.context, CGRectMake(x,y,width,height))

    def clip_to_rects(self, object rects):
        """
        """
        cdef int n
        n = len(rects)
        cdef int i
        cdef CGRect* cgrects

        cgrects = <CGRect*>PyMem_Malloc(n*sizeof(CGRect))
        if cgrects == NULL:
            raise MemoryError("could not allocate memory for CGRects")

        for i from 0 <= i < n:
            cgrects[i] = CGRectMakeFromPython(rects[i])
        CGContextClipToRects(self.context, cgrects, n)
        PyMem_Free(cgrects)


    #----------------------------------------------------------------
    # Color space manipulation
    #
    # I'm not sure we'll mess with these at all.  They seem to
    # be for setting the color system.  Hard coding to RGB or
    # RGBA for now sounds like a reasonable solution.
    #----------------------------------------------------------------

    def set_fill_color_space(self):
        """
        """
        msg = "set_fill_color_space not implemented on Macintosh yet."
        raise NotImplementedError(msg)

    def set_stroke_color_space(self):
        """
        """
        msg = "set_stroke_color_space not implemented on Macintosh yet."
        raise NotImplementedError(msg)

    def set_rendering_intent(self, intent):
        """
        """
        CGContextSetRenderingIntent(self.context, intent)

    #----------------------------------------------------------------
    # Color manipulation
    #----------------------------------------------------------------

    def set_fill_color(self, object color):
        """
        """
        r,g,b = color[:3]
        try:
            a = color[3]
        except IndexError:
            a = 1.0
        self.fill_color = (r,g,b,a)
        CGContextSetRGBFillColor(self.context, r, g, b, a)

    def set_stroke_color(self, object color):
        """
        """
        r,g,b = color[:3]
        try:
            a = color[3]
        except IndexError:
            a = 1.0
        self.stroke_color = (r,g,b,a)
        CGContextSetRGBStrokeColor(self.context, r, g, b, a)

    def set_alpha(self, float alpha):
        """
        """
        CGContextSetAlpha(self.context, alpha)

    #def set_gray_fill_color(self):
    #    """
    #    """
    #    pass

    #def set_gray_stroke_color(self):
    #    """
    #    """
    #    pass

    #def set_rgb_fill_color(self):
    #    """
    #    """
    #    pass

    #def set_rgb_stroke_color(self):
    #    """
    #    """
    #    pass

    #def cmyk_fill_color(self):
    #    """
    #    """
    #    pass

    #def cmyk_stroke_color(self):
    #    """
    #    """
    #    pass

    #----------------------------------------------------------------
    # Drawing Images
    #----------------------------------------------------------------

    def draw_image(self, object image, object rect=None):
        """ Draw an image or another CGContext onto a region.
        """
        from PIL import Image

        if rect is None:
            rect = (0, 0, self.width(), self.height())
        if isinstance(image, numpy.ndarray):
            self._draw_cgimage(CGImage(image), rect)
        elif isinstance(image, Image.Image):
            self._draw_cgimage(CGImage(numpy.array(image)), rect)
        elif isinstance(image, CGImage):
            self._draw_cgimage(image, rect)
        elif hasattr(image, 'bmp_array'):
            self._draw_cgimage(CGImage(image.bmp_array), rect)
        elif isinstance(image, CGLayerContext):
            self._draw_cglayer(image, rect)
        else:
            raise TypeError("could not recognize image %r" % type(image))

    def _draw_cgimage(self, CGImage image, object rect):
        """ Draw a CGImage into a region.
        """
        CGContextDrawImage(self.context, CGRectMakeFromPython(rect),
            image.image)

    def _draw_cglayer(self, CGLayerContext layer, object rect):
        """ Draw a CGLayer into a region.
        """
        CGContextDrawLayerInRect(self.context, CGRectMakeFromPython(rect),
            layer.layer)

    def set_interpolation_quality(self, quality):
        CGContextSetInterpolationQuality(self.context, quality)

    #----------------------------------------------------------------
    # Drawing PDF documents
    #----------------------------------------------------------------

    def draw_pdf_document(self, object rect, CGPDFDocument document not None,
        int page=1):
        """
            rect:(x,y,width,height) -- rectangle to draw into
            document:CGPDFDocument -- PDF file to read from
            page=1:int -- page number of PDF file
        """
        cdef CGRect cgrect
        cgrect = CGRectMakeFromPython(rect)

        CGContextDrawPDFDocument(self.context, cgrect, document.document, page)


    #----------------------------------------------------------------
    # Drawing Text
    #----------------------------------------------------------------

    def select_font(self, object name, float size, style='regular', encoding=None):
        """ Select a font to use.
        """
        key = (name, size, style)
        if key not in self.font_cache:
            self.current_style = default_font_info.lookup(name, style=style)
            self.font_cache[key] = self.current_style.get_font(size)

        self.current_font = self.font_cache[key]

    def set_font(self, font):
        """ Set the font for the current graphics context.

            I need to figure out this one.
        """

        style = {
            constants.NORMAL: 'regular',
            constants.BOLD: 'bold',
            constants.ITALIC: 'italic',
            constants.BOLD_ITALIC: 'bold italic',
        }[font.weight | font.style]
        self.select_font(font.face_name, font.size, style=style)

    def set_font_size(self, float size):
        """ Change the size of the currently selected font
        """
        if self.current_style is None:
            return

        name = self.current_style.family_name
        style = self.current_style.style
        key = (name, size, style)
        if key not in self.font_cache:
            self.font_cache[key] = self.current_style.get_font(size)

        self.current_font = self.font_cache[key]

    def set_character_spacing(self, float spacing):
        """ Set the spacing between characters when drawing text
        """
        # XXX: Perhaps this should be handled by the kerning attribute
        # in the attributed string that is used to make a line of text?
        CGContextSetCharacterSpacing(self.context, spacing)

    def get_character_spacing(self):
        """ Get the current spacing between characters when drawing text
        """
        # XXX: There is no "get" counterpart for CGContextSetCharacterSpacing
        msg = "get_character_spacing not implemented for Quartz yet."
        raise NotImplementedError(msg)

    def set_text_drawing_mode(self, object mode):
        """
        """
        try:
            cgmode = text_modes[mode]
        except KeyError:
            msg = "Invalid text drawing mode.  See documentation for valid modes"
            raise ValueError(msg)
        CGContextSetTextDrawingMode(self.context, cgmode)

    def set_text_position(self, float x,float y):
        """
        """
        self.text_matrix.tx = x
        self.text_matrix.ty = y

    def get_text_position(self):
        """
        """
        return self.text_matrix.tx, self.text_matrix.ty

    def set_text_matrix(self, object ttm):
        """
        """
        cdef float a,b,c,d,tx,ty

        # Handle both matrices that this class returns and agg._AffineMatrix
        # instances.
        try:
            ((a,  b,  _),
             (c,  d,  _),
             (tx, ty, _)) = ttm
        except:
            a,b,c,d,tx,ty = ttm

        cdef CGAffineTransform transform
        transform = CGAffineTransformMake(a,b,c,d,tx,ty)
        self.text_matrix = transform

    def get_text_matrix(self):
        """
        """
        return ((self.text_matrix.a, self.text_matrix.b, 0.0),
                (self.text_matrix.c, self.text_matrix.d, 0.0),
                (self.text_matrix.tx,self.text_matrix.ty,1.0))

    def get_text_extent(self, object text):
        """ Measure the space taken up by given text using the current font.
        """
        cdef size_t pointer
        cdef CTFontRef ct_font
        cdef CTLineRef ct_line
        cdef CGFloat ascent = 0.0, descent = 0.0
        cdef double x1,x2,y1,y2, width = 0.0

        pointer = self.current_font.get_pointer()
        ct_font = <CTFontRef>pointer
        ct_line = _create_ct_line(text, ct_font, None)
        if ct_line != NULL:
            width = CTLineGetTypographicBounds(ct_line, &ascent, &descent, NULL)
            CFRelease(ct_line)

        x1 = 0.0
        x2 = width
        y1 = -descent
        y2 = -y1 + ascent

        return x1, y1, x2, y2

    def get_full_text_extent(self, object text):
        """ Backwards compatibility API over .get_text_extent() for Enable.
        """

        x1, y1, x2, y2 = self.get_text_extent(text)

        return x2, y2, y1, x1


    def show_text(self, object text, object xy=None):
        """ Draw text on the device at current text position.

            This is also used for showing text at a particular point
            specified by xy == (x, y).
        """
        cdef float x, y
        cdef size_t pointer
        cdef CTFontRef ct_font
        cdef CTLineRef ct_line

        if not text:
            # Nothing to draw
            return

        pointer = self.current_font.get_pointer()
        ct_font = <CTFontRef>pointer
        ct_line = _create_ct_line(text, ct_font, self.stroke_color)
        if ct_line == NULL:
            return

        if xy is None:
            x = 0.0
            y = 0.0
        else:
            x = xy[0]
            y = xy[1]

        self.save_state()
        try:
            CGContextConcatCTM(self.context, self.text_matrix)
            CGContextSetTextPosition(self.context, x, y)
            CTLineDraw(ct_line, self.context)
        finally:
            self.restore_state()
            CFRelease(ct_line)

    def show_text_at_point(self, object text, float x, float y):
        """ Draw text on the device at a given text position.
        """
        self.show_text(text, (x, y))

    def show_glyphs(self):
        """
        """
        msg = "show_glyphs not implemented on Macintosh yet."
        raise NotImplementedError(msg)

    #----------------------------------------------------------------
    # Painting paths (drawing and filling contours)
    #----------------------------------------------------------------

    def stroke_path(self):
        """
        """
        CGContextStrokePath(self.context)

    def fill_path(self):
        """
        """
        CGContextFillPath(self.context)

    def eof_fill_path(self):
        """
        """
        CGContextEOFillPath(self.context)

    def stroke_rect(self, object rect):
        """
        """
        CGContextStrokeRect(self.context, CGRectMakeFromPython(rect))

    def stroke_rect_with_width(self, object rect, float width):
        """
        """
        CGContextStrokeRectWithWidth(self.context, CGRectMakeFromPython(rect), width)

    def fill_rect(self, object rect):
        """
        """
        CGContextFillRect(self.context, CGRectMakeFromPython(rect))

    def fill_rects(self, object rects):
        """
        """
        cdef int n
        n = len(rects)
        cdef int i
        cdef CGRect* cgrects

        cgrects = <CGRect*>PyMem_Malloc(n*sizeof(CGRect))
        if cgrects == NULL:
            raise MemoryError("could not allocate memory for CGRects")

        for i from 0 <= i < n:
            cgrects[i] = CGRectMakeFromPython(rects[i])

        CGContextFillRects(self.context, cgrects, n)

    def clear_rect(self, object rect):
        """
        """
        CGContextClearRect(self.context, CGRectMakeFromPython(rect))

    def draw_path(self, object mode=constants.FILL_STROKE):
        """ Walk through all the drawing subpaths and draw each element.

            Each subpath is drawn separately.
        """

        cg_mode = draw_modes[mode]
        CGContextDrawPath(self.context, cg_mode)

    def draw_rect(self, rect, object mode=constants.FILL_STROKE):
        """ Draw a rectangle with the given mode.
        """

        self.save_state()
        CGContextBeginPath(self.context)
        CGContextAddRect(self.context, CGRectMakeFromPython(rect))
        cg_mode = draw_modes[mode]
        CGContextDrawPath(self.context, cg_mode)
        self.restore_state()

    def get_empty_path(self):
        """ Return a path object that can be built up and then reused.
        """

        return CGMutablePath()

    def draw_path_at_points(self, points, CGMutablePath marker not None,
        object mode=constants.FILL_STROKE):

        cdef int i
        cdef int n
        cdef c_numpy.ndarray apoints
        cdef float x, y

        apoints = <c_numpy.ndarray>(numpy.asarray(points, dtype=numpy.float32))

        if apoints.nd != 2 or apoints.dimensions[1] != 2:
            msg = "must pass array of 2-D points"
            raise ValueError(msg)

        cg_mode = draw_modes[mode]

        n = len(points)
        for i from 0 <= i < n:
            x = (<float*>c_numpy.PyArray_GETPTR2(apoints, i, 0))[0]
            y = (<float*>c_numpy.PyArray_GETPTR2(apoints, i, 1))[0]
            CGContextSaveGState(self.context)
            CGContextTranslateCTM(self.context, x, y)
            CGContextAddPath(self.context, marker.path)
            CGContextDrawPath(self.context, cg_mode)
            CGContextRestoreGState(self.context)

    def linear_gradient(self, x1, y1, x2, y2, stops, spread_method, units='userSpaceOnUse'):
        cdef CGRect path_rect
        if units == 'objectBoundingBox':
            # transform from relative coordinates
            path_rect = CGContextGetPathBoundingBox(self.context)
            x1 = path_rect.origin.x + x1 * path_rect.size.width
            x2 = path_rect.origin.x + x2 * path_rect.size.width
            y1 = path_rect.origin.y + y1 * path_rect.size.height
            y2 = path_rect.origin.y + y2 * path_rect.size.height

        stops_list = stops.transpose().tolist()
        func = PiecewiseLinearColorFunction(stops_list)

        # Shadings fill the current clip path
        self.clip()
        if spread_method == 'pad' or spread_method == '':
            shading = AxialShading(func, (x1,y1), (x2,y2),
                                   extend_start=1, extend_end=1)
            self.draw_shading(shading)
        else:
            self.repeat_linear_shading(x1, y1, x2, y2, stops, spread_method, func)

    def repeat_linear_shading(self, x1, y1, x2, y2, stops, spread_method,
                              ShadingFunction func not None):
        cdef CGRect clip_rect = CGContextGetClipBoundingBox(self.context)
        cdef double dirx, diry, slope
        cdef double startx, starty, endx, endy
        cdef int func_index = 0

        if spread_method == 'reflect':
            # generate the mirrored color function
            stops_list = stops[::-1].transpose()
            stops_list[0] = 1-stops_list[0]
            funcs = [func, PiecewiseLinearColorFunction(stops_list.tolist())]
        else:
            funcs = [func, func]

        dirx, diry = x2-x1, y2-y1
        startx, starty = x1, y1
        endx, endy = x2, y2
        if dirx == 0. and diry == 0.:
            slope = float('nan')
            diry = 100.0
        elif diry == 0.:
            slope = 0.
        else:
            # perpendicular slope
            slope = -dirx/diry
        while _line_intersects_cgrect(startx, starty, slope, clip_rect):
            shading = AxialShading(funcs[func_index&1],
                                   (startx, starty), (endx, endy),
                                   extend_start=0, extend_end=0)
            self.draw_shading(shading)
            startx, starty = endx, endy
            endx, endy = endx+dirx, endy+diry
            func_index += 1

        # reverse direction
        dirx, diry = x1-x2, y1-y2
        startx, starty = x1+dirx, y1+diry
        endx, endy = x1, y1
        func_index = 1
        if dirx == 0. and diry == 0.:
            slope = float('nan')
            diry = 100.0
        elif diry == 0.:
            slope = 0.
        else:
            # perpendicular slope
            slope = -dirx/diry
        while _line_intersects_cgrect(endx, endy, slope, clip_rect):
            shading = AxialShading(funcs[func_index&1],
                                   (startx, starty), (endx, endy),
                                   extend_start=0, extend_end=0)
            self.draw_shading(shading)
            endx, endy = startx, starty
            startx, starty = endx+dirx, endy+diry
            func_index += 1

    def radial_gradient(self, cx, cy, r, fx, fy,  stops, spread_method, units='userSpaceOnUse'):
        cdef CGRect path_rect
        if units == 'objectBoundingBox':
            # transform from relative coordinates
            path_rect = CGContextGetPathBoundingBox(self.context)
            r = r * path_rect.size.width
            cx = path_rect.origin.x + cx * path_rect.size.width
            fx = path_rect.origin.x + fx * path_rect.size.width
            cy = path_rect.origin.y + cy * path_rect.size.height
            fy = path_rect.origin.y + fy * path_rect.size.height

        stops_list = stops.transpose().tolist()
        func = PiecewiseLinearColorFunction(stops_list)

        # 12/11/2010 John Wiggins - In order to avoid a bug in Quartz,
        # the radius of a radial gradient must be greater than the distance
        # between the center and the focus. When input runs afoul of this bug,
        # the focus point will be moved towards the center so that the distance
        # is less than the radius. This matches the behavior of AGG.
        cdef double dx = fx-cx
        cdef double dy = fy-cy
        cdef double dist = sqrt(dx*dx + dy*dy)
        if r <= dist:
            newdist = r-0.001
            fx = cx+newdist*dx/dist
            fy = cy+newdist*dy/dist

        # Shadings fill the current clip path
        self.clip()
        if spread_method == 'pad' or spread_method == '':
            shading = RadialShading(func, (fx, fy), 0.0, (cx, cy), r,
                                   extend_start=1, extend_end=1)
            self.draw_shading(shading)
        else:
            # 'reflect' and 'repeat' need to iterate
            self.repeat_radial_shading(cx, cy, r, fx, fy, stops, spread_method, func)

    def repeat_radial_shading(self, cx, cy, r, fx, fy, stops, spread_method,
                              ShadingFunction func not None):
        cdef CGRect clip_rect = CGContextGetClipBoundingBox(self.context)
        cdef double rad = 0., dirx = 0., diry = 0.
        cdef double startx, starty, endx, endy
        cdef int func_index = 0

        if spread_method == 'reflect':
            # generate the mirrored color function
            stops_list = stops[::-1].transpose()
            stops_list[0] = 1-stops_list[0]
            funcs = [func, PiecewiseLinearColorFunction(stops_list.tolist())]
        else:
            funcs = [func, func]

        dirx, diry = cx-fx, cy-fy
        startx, starty = fx,fy
        endx, endy = cx, cy
        while not _cgrect_within_circle(clip_rect, endx, endy, rad):
            shading = RadialShading(funcs[func_index & 1],
                                    (startx, starty), rad, (endx, endy), rad+r,
                                    extend_start=0, extend_end=0)
            self.draw_shading(shading)
            startx, starty = endx, endy
            endx, endy = endx+dirx, endy+diry
            rad += r
            func_index += 1

    def draw_shading(self, Shading shading not None):
        CGContextDrawShading(self.context, shading.shading)


    #----------------------------------------------------------------
    # Extra routines that aren't part of DisplayPDF
    #
    # Some access to font metrics are needed for laying out text.
    # Not sure how to handle this yet.  The candidates below are
    # from Piddle.  Perhaps there is another alternative?
    #
    #----------------------------------------------------------------

    #def font_height(self):
    #    '''Find the total height (ascent + descent) of the given font.'''
    #    #return self.font_ascent() + self.font_descent()

    #def font_ascent(self):
    #    '''Find the ascent (height above base) of the given font.'''
    #    pass

    #def font_descent(self):
    #    '''Find the descent (extent below base) of the given font.'''
    #    extents = self.dc.GetFullTextExtent(' ', wx_font)
    #    return extents[2]

    def __dealloc__(self):
        if self.context != NULL and self.can_release:
            CGContextRelease(self.context)
            self.context = NULL

    # The following are Quartz APIs not in Kiva

    def set_pattern_phase(self, float tx, float ty):
        """
            tx,ty:floats -- A translation in user-space to apply to a
                           pattern before it is drawn
        """
        CGContextSetPatternPhase(self.context, CGSizeMake(tx, ty))

    def set_should_smooth_fonts(self, bool value):
        """
            value:bool -- specify whether to enable font smoothing or not
        """
        CGContextSetShouldSmoothFonts(self.context, value)

cdef class CGContextInABox(CGContext):
    """ A CGContext that knows its size.
    """
    cdef readonly object size
    cdef readonly int _width
    cdef readonly int _height

    def __init__(self, object size, size_t context, long can_release=0,
                 *args, **kwargs):
        self.context = <CGContextRef>context

        self.can_release = can_release

        self._width, self._height = size

        self._setup_color_space()
        self._setup_fonts()

    def clear(self, object clear_color=(1.0,1.0,1.0,1.0)):
        self.save_state()
        # Reset the transformation matrix back to the identity.
        CGContextConcatCTM(self.context,
            CGAffineTransformInvert(CGContextGetCTM(self.context)))
        self.set_fill_color(clear_color)
        CGContextFillRect(self.context, CGRectMake(0,0,self._width,self._height))
        self.restore_state()

    def width(self):
        return self._width

    def height(self):
        return self._height


cdef class CGLayerContext(CGContextInABox):
    cdef CGLayerRef layer
    cdef object gc

    def __init__(self, object size, CGContext gc not None, *args, **kwargs):
        self.gc = <object>gc
        self.layer = CGLayerCreateWithContext(gc.context,
            CGSizeMake(size[0], size[1]), NULL)
        self.context = CGLayerGetContext(self.layer)
        self.size = size
        self._width, self._height = size
        self.can_release = 1

        self._setup_color_space()
        self._setup_fonts()

    def __dealloc__(self):
        if self.layer != NULL:
            CGLayerRelease(self.layer)
            self.layer = NULL
            # The documentation doesn't say whether I need to release the
            # context derived from the layer or not. I believe that means
            # I don't.
            self.context = NULL
        self.gc = None

    def save(self, object filename, file_format=None, pil_options=None):
        """ Save the GraphicsContext to a file.  Output files are always saved
        in RGB or RGBA format; if this GC is not in one of these formats, it is
        automatically converted.

        If filename includes an extension, the image format is inferred from it.
        file_format is only required if the format can't be inferred from the
        filename (e.g. if you wanted to save a PNG file as a .dat or .bin).

        filename may also be "file-like" object such as a StringIO, in which
        case a file_format must be supplied.

        pil_options is a dict of format-specific options that are passed down to
        the PIL image file writer.  If a writer doesn't recognize an option, it
        is silently ignored.

        If the image has an alpha channel and the specified output file format
        does not support alpha, the image is saved in rgb24 format.
        """

        cdef CGBitmapContext bmp

        # Create a CGBitmapContext from this layer, draw to it, then let it save
        # itself out.
        rect = (0, 0) + self.size
        bmp = CGBitmapContext(self.size, base_pixel_scale=self.base_scale)
        CGContextDrawLayerInRect(bmp.context,  CGRectMakeFromPython(rect), self.layer)
        bmp.save(filename, file_format=file_format, pil_options=pil_options)


cdef class CGContextFromSWIG(CGContext):
    def __init__(self, swig_obj):
        self.can_release = False
        ptr = int(swig_obj.this.split('_')[1], 16)
        CGContext.__init__(self, ptr)


cdef class CGPDFContext(CGContext):
    cdef readonly char* filename
    cdef CGRect media_box
    def __init__(self, char* filename, rect=None):
        cdef CFURLRef cfurl
        cfurl = url_from_filename(filename)
        cdef CGRect cgrect
        cdef CGRect* cgrect_ptr

        if rect is None:
            cgrect = CGRectMake(0,0,612,792)
            cgrect_ptr = &cgrect
        else:
            cgrect = CGRectMakeFromPython(rect)
            cgrect_ptr = &cgrect
        self.context = CGPDFContextCreateWithURL(cfurl, cgrect_ptr, NULL)
        CFRelease(cfurl)

        self.filename = filename
        self.media_box = cgrect

        if self.context == NULL:
            raise RuntimeError("could not create CGPDFContext")
        self.can_release = 1

        self._setup_color_space()
        self._setup_fonts()

        CGContextBeginPage(self.context, cgrect_ptr)

    def begin_page(self, media_box=None):
        cdef CGRect* box_ptr
        cdef CGRect box
        if media_box is None:
            box_ptr = &(self.media_box)
        else:
            box = CGRectMakeFromPython(media_box)
            box_ptr = &box
        CGContextBeginPage(self.context, box_ptr)

    def flush(self, end_page=True):
        if end_page:
            self.end_page()
        CGContextFlush(self.context)

    def begin_transparency_layer(self):
        CGContextBeginTransparencyLayer(self.context, NULL)

    def end_transparency_layer(self):
        CGContextEndTransparencyLayer(self.context)

cdef class CGBitmapContext(CGContext):
    cdef void* data

    def __cinit__(self, *args, **kwds):
        self.data = NULL

    def __init__(self, object size_or_array, bool grey_scale=0,
        int bits_per_component=8, int bytes_per_row=-1,
        alpha_info=kCGImageAlphaPremultipliedLast, base_pixel_scale=1.0):

        cdef int bits_per_pixel
        cdef CGColorSpaceRef colorspace
        cdef void* dataptr

        self.base_scale = base_pixel_scale

        if hasattr(size_or_array, '__array_interface__'):
            # It's an array.
            arr = numpy.asarray(size_or_array, order='C')
            typestr = arr.dtype.str
            if typestr != '|u1':
                raise ValueError("expecting an array of unsigned bytes; got %r"
                    % typestr)
            shape = arr.shape
            if len(shape) != 3 or shape[-1] not in (3, 4):
                raise ValueError("expecting a shape (width, height, depth) "
                    "with depth either 3 or 4; got %r" % shape)
            height, width, depth = shape
            if depth == 3:
                # Need to add an alpha channel.
                alpha = numpy.empty((height, width), dtype=numpy.uint8)
                alpha.fill(255)
                arr = numpy.dstack([arr, alpha])
                depth = 4
            ptr, readonly = arr.__array_interface__['data']
            dataptr = <void*><size_t>ptr
        else:
            # It's a size tuple.
            width, height = size_or_array
            arr = None

        if grey_scale:
            alpha_info = kCGImageAlphaNone
            bits_per_component = 8
            bits_per_pixel = 8
            colorspace = CGColorSpaceCreateWithName(kCGColorSpaceGenericGray)
        elif bits_per_component == 5:
            alpha_info = kCGImageAlphaNoneSkipFirst
            bits_per_pixel = 16
            colorspace = CGColorSpaceCreateWithName(kCGColorSpaceGenericRGB)
        elif bits_per_component == 8:
            if alpha_info not in (kCGImageAlphaNoneSkipFirst,
                                  kCGImageAlphaNoneSkipLast,
                                  kCGImageAlphaPremultipliedFirst,
                                  kCGImageAlphaPremultipliedLast,
                                 ):
                raise ValueError("not a valid alpha_info")
            bits_per_pixel = 32
            colorspace = CGColorSpaceCreateWithName(kCGColorSpaceGenericRGB)
        else:
            raise ValueError("bits_per_component must be 5 or 8")

        cdef int min_bytes
        min_bytes = (width*bits_per_pixel + 7) / 8
        if bytes_per_row < min_bytes:
            bytes_per_row = min_bytes

        self.data = PyMem_Malloc(height*bytes_per_row)
        if self.data == NULL:
            CGColorSpaceRelease(colorspace)
            raise MemoryError("could not allocate memory")
        if arr is not None:
            # Copy the data from the array.
            memcpy(self.data, dataptr, width*height*depth)

        self.context = CGBitmapContextCreate(self.data, width, height,
            bits_per_component, bytes_per_row, colorspace, alpha_info)
        CGColorSpaceRelease(colorspace)

        if self.context == NULL:
            raise RuntimeError("could not create CGBitmapContext")
        self.can_release = 1

        self._setup_fonts()


    def __dealloc__(self):
        if self.context != NULL and self.can_release:
            CGContextRelease(self.context)
            self.context = NULL
        if self.data != NULL:
            # Hmm, this could be tricky if anything in Quartz retained a
            # reference to self.context
            PyMem_Free(self.data)
            self.data = NULL

    property alpha_info:
        def __get__(self):
            return CGBitmapContextGetAlphaInfo(self.context)

    property bits_per_component:
        def __get__(self):
            return CGBitmapContextGetBitsPerComponent(self.context)

    property bits_per_pixel:
        def __get__(self):
            return CGBitmapContextGetBitsPerPixel(self.context)

    property bytes_per_row:
        def __get__(self):
            return CGBitmapContextGetBytesPerRow(self.context)

#    property colorspace:
#        def __get__(self):
#            return CGBitmapContextGetColorSpace(self.context)

    def height(self):
        return CGBitmapContextGetHeight(self.context)

    def width(self):
        return CGBitmapContextGetWidth(self.context)

    def __getbuffer__(self, Py_buffer *buffer, int flags):
        """ When another object calls PyObject_GetBuffer on us.
        """
        cdef Py_ssize_t* shape = <Py_ssize_t *>PyMem_Malloc(2 * sizeof(Py_ssize_t))
        cdef Py_ssize_t* strides = <Py_ssize_t *>PyMem_Malloc(2 * sizeof(Py_ssize_t))

        shape[0] = self.bytes_per_row
        shape[1] = self.height()
        strides[0] = self.bytes_per_row
        strides[1] = 1

        buffer.buf = <char *>(self.data)
        buffer.obj = self
        buffer.len = self.height() * self.bytes_per_row
        buffer.readonly = 1
        buffer.itemsize = 1
        buffer.format = 'b'
        buffer.ndim = 2
        buffer.shape = shape
        buffer.strides = strides
        buffer.suboffsets = NULL
        buffer.internal = NULL

    def __releasebuffer__(self, Py_buffer *buffer):
        """ When PyBuffer_Release is called on buffers from __getbuffer__.
        """
        # Just deallocate the shape and strides allocated by __getbuffer__.
        # Since buffer.obj is a referenced counted reference to this object
        # (thus keeping this object alive as long a connected buffer exists)
        # and we don't mutate `self.data` outside of __init__ and __dealloc__,
        # we have nothing further to do here.
        PyMem_Free(buffer.shape)
        PyMem_Free(buffer.strides)

    def clear(self, object clear_color=(1.0, 1.0, 1.0, 1.0)):
        """Paint over the whole image with a solid color.
        """

        self.save_state()
        # Reset the transformation matrix back to the identity.
        CGContextConcatCTM(self.context,
            CGAffineTransformInvert(CGContextGetCTM(self.context)))

        self.set_fill_color(clear_color)
        CGContextFillRect(self.context, CGRectMake(0, 0, self.width(), self.height()))
        self.restore_state()

    def save(self, object filename, file_format=None, pil_options=None):
        """ Save the GraphicsContext to a file.  Output files are always saved
        in RGB or RGBA format; if this GC is not in one of these formats, it is
        automatically converted.

        If filename includes an extension, the image format is inferred from it.
        file_format is only required if the format can't be inferred from the
        filename (e.g. if you wanted to save a PNG file as a .dat or .bin).

        filename may also be "file-like" object such as a StringIO, in which
        case a file_format must be supplied.

        pil_options is a dict of format-specific options that are passed down to
        the PIL image file writer.  If a writer doesn't recognize an option, it
        is silently ignored.

        If the image has an alpha channel and the specified output file format
        does not support alpha, the image is saved in rgb24 format.
        """

        try:
            from PIL import Image as PilImage
        except ImportError:
            raise ImportError("need Pillow to save images")

        if self.bits_per_pixel == 32:
            if self.alpha_info == kCGImageAlphaPremultipliedLast:
                mode = 'RGBA'
            elif self.alpha_info == kCGImageAlphaPremultipliedFirst:
                mode = 'ARGB'
            else:
                raise ValueError("cannot save this pixel format")
        elif self.bits_per_pixel == 8:
            mode = 'L'
        else:
            raise ValueError("cannot save this pixel format")

        if file_format is None:
            file_format = ''
        if pil_options is None:
            pil_options = {}

        file_ext = (
            os.path.splitext(filename)[1][1:] if isinstance(filename, str)
            else ''
        )

        # Check te output format to see if DPI can be passed
        dpi_formats = ('jpg', 'png', 'tiff', 'jpeg')
        if file_ext in dpi_formats or file_format.lower() in dpi_formats:
            # Assume 72dpi is 1x
            dpi = int(72 * self.base_scale)
            pil_options['dpi'] = (dpi, dpi)

        img = PilImage.frombuffer(mode, (self.width(), self.height()), self,
                                  'raw', mode, 0, 1)
        if 'A' in mode:
            # Check the output format to see if it can handle an alpha channel.
            no_alpha_formats = ('jpg', 'bmp', 'eps', 'jpeg')
            if (file_ext in no_alpha_formats or
                    file_format.lower() in no_alpha_formats):
                img = img.convert('RGB')

        img.save(filename, format=file_format, **pil_options)

cdef class CGImage:
    cdef CGImageRef image
    cdef void* data
    cdef readonly c_numpy.ndarray bmp_array

    def __cinit__(self, *args, **kwds):
        self.image = NULL

    property width:
        def __get__(self):
            return CGImageGetWidth(self.image)

    property height:
        def __get__(self):
            return CGImageGetHeight(self.image)

    property bits_per_component:
        def __get__(self):
            return CGImageGetBitsPerComponent(self.image)

    property bits_per_pixel:
        def __get__(self):
            return CGImageGetBitsPerPixel(self.image)

    property bytes_per_row:
        def __get__(self):
            return CGImageGetBytesPerRow(self.image)

    property alpha_info:
        def __get__(self):
            return CGImageGetAlphaInfo(self.image)

    property should_interpolate:
        def __get__(self):
            return CGImageGetShouldInterpolate(self.image)

    property is_mask:
        def __get__(self):
            return CGImageIsMask(self.image)

    def __init__(self, object size_or_array, bool grey_scale=0,
        int bits_per_component=8, int bytes_per_row=-1,
        alpha_info=kCGImageAlphaLast, int should_interpolate=1):

        cdef int bits_per_pixel
        cdef CGColorSpaceRef colorspace

        if hasattr(size_or_array, '__array_interface__'):
            # It's an array.
            arr = size_or_array
            typestr = arr.__array_interface__['typestr']
            if typestr != '|u1':
                raise ValueError("expecting an array of unsigned bytes; got %r"
                    % typestr)
            shape = arr.__array_interface__['shape']
            if grey_scale:
                if (len(shape) == 3 and shape[-1] != 1) or (len(shape) != 2):
                    raise ValueError("with grey_scale, expecting a shape "
                        "(height, width) or (height, width, 1); got "
                        "%r" % (shape,))
                height, width = shape[:2]
                depth = 1
            else:
                if len(shape) != 3 or shape[-1] not in (3, 4):
                    raise ValueError("expecting a shape (height, width, depth) "
                        "with depth either 3 or 4; got %r" % (shape,))
                height, width, depth = shape
            if depth in (1, 3):
                alpha_info = kCGImageAlphaNone
            else:
                # Make a copy.
                arr = numpy.array(arr)
                alpha_info = kCGImageAlphaPremultipliedLast
        else:
            # It's a size tuple.
            width, height = size_or_array
            if grey_scale:
                lastdim = 1
                alpha_info = kCGImageAlphaNone
            else:
                lastdim = 4
                alpha_info = kCGImageAlphaPremultipliedLast
            arr = numpy.zeros((height, width, lastdim), dtype=numpy.uint8)

        self.bmp_array = <c_numpy.ndarray>arr
        Py_INCREF(self.bmp_array)
        self.data = <void*>c_numpy.PyArray_DATA(self.bmp_array)

        if grey_scale:
            alpha_info = kCGImageAlphaNone
            bits_per_component = 8
            bits_per_pixel = 8
            colorspace = CGColorSpaceCreateDeviceGray()
        elif bits_per_component == 5:
            alpha_info = kCGImageAlphaNoneSkipFirst
            bits_per_pixel = 16
            colorspace = CGColorSpaceCreateDeviceRGB()
        elif bits_per_component == 8:
            if alpha_info in (kCGImageAlphaNoneSkipFirst,
                              kCGImageAlphaNoneSkipLast,
                              kCGImageAlphaPremultipliedFirst,
                              kCGImageAlphaPremultipliedLast,
                              kCGImageAlphaFirst,
                              kCGImageAlphaLast,
                             ):
                bits_per_pixel = 32
            elif alpha_info == kCGImageAlphaNone:
                bits_per_pixel = 24
            colorspace = CGColorSpaceCreateDeviceRGB()
        else:
            raise ValueError("bits_per_component must be 5 or 8")

        cdef int min_bytes
        min_bytes = (width*bits_per_pixel + 7) / 8
        if bytes_per_row < min_bytes:
            bytes_per_row = min_bytes

        cdef CGDataProviderRef provider
        provider = CGDataProviderCreateWithData(
            NULL, self.data, c_numpy.PyArray_SIZE(self.bmp_array), NULL)
        if provider == NULL:
            raise RuntimeError("could not make provider")

        cdef CGColorSpaceRef space
        space = CGColorSpaceCreateDeviceRGB()

        self.image = CGImageCreate(width, height, bits_per_component,
            bits_per_pixel, bytes_per_row, space, alpha_info, provider, NULL,
            should_interpolate, kCGRenderingIntentDefault)
        CGColorSpaceRelease(space)
        CGDataProviderRelease(provider)

        if self.image == NULL:
            raise RuntimeError("could not make image")

    def __dealloc__(self):
        if self.image != NULL:
            CGImageRelease(self.image)
            self.image = NULL
        Py_XDECREF(self.bmp_array)

cdef class CGImageFile(CGImage):
    def __init__(self, object image_or_filename, int should_interpolate=1):
        cdef int width, height, bits_per_component, bits_per_pixel, bytes_per_row
        cdef CGImageAlphaInfo alpha_info

        from PIL import Image
        import types

        if type(image_or_filename) is str:
            img = Image.open(image_or_filename)
            img.load()
        elif isinstance(image_or_filename, Image.Image):
            img = image_or_filename
        else:
            raise ValueError("need a PIL Image or a filename")

        width, height = img.size
        mode = img.mode

        if mode not in ["L", "RGB","RGBA"]:
            img = img.convert(mode="RGBA")
            mode = 'RGBA'

        bits_per_component = 8

        if mode == 'RGB':
            bits_per_pixel = 24
            alpha_info = kCGImageAlphaNone
        elif mode == 'RGBA':
            bits_per_pixel = 32
            alpha_info = kCGImageAlphaPremultipliedLast
        elif mode == 'L':
            bits_per_pixel = 8
            alpha_info = kCGImageAlphaNone

        bytes_per_row = (bits_per_pixel*width + 7)/ 8

        cdef char* data
        cdef char* py_data
        cdef int dims[3]
        dims[0] = height
        dims[1] = width
        dims[2] = bits_per_pixel/bits_per_component

        self.bmp_array = c_numpy.PyArray_SimpleNew(3, &(dims[0]), c_numpy.NPY_UBYTE)

        data = self.bmp_array.data
        bs = img.tobytes()
        py_data = PyBytes_AsString(bs)

        memcpy(<void*>data, <void*>py_data, len(bs))

        self.data = data

        cdef CGDataProviderRef provider
        provider = CGDataProviderCreateWithData(
            NULL, <void*>data, len(data), NULL)

        if provider == NULL:
            raise RuntimeError("could not make provider")

        cdef CGColorSpaceRef space
        space = CGColorSpaceCreateDeviceRGB()

        self.image = CGImageCreate(width, height, bits_per_component,
            bits_per_pixel, bytes_per_row, space, alpha_info, provider, NULL,
            should_interpolate, kCGRenderingIntentDefault)
        CGColorSpaceRelease(space)
        CGDataProviderRelease(provider)

        if self.image == NULL:
            raise RuntimeError("could not make image")

    def __dealloc__(self):
        if self.image != NULL:
            CGImageRelease(self.image)
            self.image = NULL
        Py_XDECREF(self.bmp_array)

cdef class CGImageMask(CGImage):
    def __init__(self, char* data, int width, int height,
        int bits_per_component, int bits_per_pixel, int bytes_per_row,
        int should_interpolate=1):

        cdef CGDataProviderRef provider
        provider = CGDataProviderCreateWithData(
            NULL, <void*>data, len(data), NULL)

        if provider == NULL:
            raise RuntimeError("could not make provider")

        self.image = CGImageMaskCreate(width, height, bits_per_component,
            bits_per_pixel, bytes_per_row, provider, NULL,
            should_interpolate)
        CGDataProviderRelease(provider)

        if self.image == NULL:
            raise RuntimeError("could not make image")

cdef class CGPDFDocument:
    cdef CGPDFDocumentRef document

    property number_of_pages:
        def __get__(self):
            return CGPDFDocumentGetNumberOfPages(self.document)

    property allows_copying:
        def __get__(self):
            return CGPDFDocumentAllowsCopying(self.document)

    property allows_printing:
        def __get__(self):
            return CGPDFDocumentAllowsPrinting(self.document)

    property is_encrypted:
        def __get__(self):
            return CGPDFDocumentIsEncrypted(self.document)

    property is_unlocked:
        def __get__(self):
            return CGPDFDocumentIsUnlocked(self.document)

    def __init__(self, char* filename):
        import os
        if not os.path.exists(filename) or not os.path.isfile(filename):
            raise ValueError("%s is not a file" % filename)

        cdef CFURLRef cfurl
        cfurl = url_from_filename(filename)

        self.document = CGPDFDocumentCreateWithURL(cfurl)
        CFRelease(cfurl)
        if self.document == NULL:
            raise RuntimeError("could not create CGPDFDocument")

    def unlock_with_password(self, char* password):
        return CGPDFDocumentUnlockWithPassword(self.document, password)

    def get_media_box(self, int page):
        cdef CGRect cgrect
        cgrect = CGPDFDocumentGetMediaBox(self.document, page)
        return (cgrect.origin.x, cgrect.origin.y,
                cgrect.size.width, cgrect.size.height)

    def get_crop_box(self, int page):
        cdef CGRect cgrect
        cgrect = CGPDFDocumentGetCropBox(self.document, page)
        return (cgrect.origin.x, cgrect.origin.y,
                cgrect.size.width, cgrect.size.height)

    def get_bleed_box(self, int page):
        cdef CGRect cgrect
        cgrect = CGPDFDocumentGetBleedBox(self.document, page)
        return (cgrect.origin.x, cgrect.origin.y,
                cgrect.size.width, cgrect.size.height)

    def get_trim_box(self, int page):
        cdef CGRect cgrect
        cgrect = CGPDFDocumentGetTrimBox(self.document, page)
        return (cgrect.origin.x, cgrect.origin.y,
                cgrect.size.width, cgrect.size.height)

    def get_art_box(self, int page):
        cdef CGRect cgrect
        cgrect = CGPDFDocumentGetArtBox(self.document, page)
        return (cgrect.origin.x, cgrect.origin.y,
                cgrect.size.width, cgrect.size.height)

    def get_rotation_angle(self, int page):
        cdef int angle
        angle = CGPDFDocumentGetRotationAngle(self.document, page)
        if angle == 0:
            raise ValueError("page %d does not exist" % page)

    def __dealloc__(self):
        if self.document != NULL:
            CGPDFDocumentRelease(self.document)
            self.document = NULL

cdef CGDataProviderRef CGDataProviderFromFilename(char* string) except NULL:
    cdef CFURLRef cfurl
    cdef CGDataProviderRef result

    cfurl = url_from_filename(string)
    if cfurl == NULL:
        raise RuntimeError("could not create CFURLRef")

    result = CGDataProviderCreateWithURL(cfurl)
    CFRelease(cfurl)
    if result == NULL:
        raise RuntimeError("could not create CGDataProviderRef")
    return result

cdef class CGAffine:
    cdef CGAffineTransform real_transform

    property a:
        def __get__(self):
            return self.real_transform.a
        def __set__(self, float value):
            self.real_transform.a = value

    property b:
        def __get__(self):
            return self.real_transform.b
        def __set__(self, float value):
            self.real_transform.b = value

    property c:
        def __get__(self):
            return self.real_transform.c
        def __set__(self, float value):
            self.real_transform.c = value

    property d:
        def __get__(self):
            return self.real_transform.d
        def __set__(self, float value):
            self.real_transform.d = value

    property tx:
        def __get__(self):
            return self.real_transform.tx
        def __set__(self, float value):
            self.real_transform.tx = value

    property ty:
        def __get__(self):
            return self.real_transform.ty
        def __set__(self, float value):
            self.real_transform.ty = value

    def __init__(self, float a=1.0, float b=0.0, float c=0.0, float d=1.0,
        float tx=0.0, float ty=0.0):
        self.real_transform = CGAffineTransformMake(a,b,c,d,tx,ty)

    def translate(self, float tx, float ty):
        self.real_transform = CGAffineTransformTranslate(self.real_transform,
            tx, ty)
        return self

    def rotate(self, float angle):
        self.real_transform = CGAffineTransformRotate(self.real_transform,
            angle)
        return self

    def scale(self, float sx, float sy):
        self.real_transform = CGAffineTransformScale(self.real_transform, sx,
            sy)
        return self

    def invert(self):
        self.real_transform = CGAffineTransformInvert(self.real_transform)
        return self

    def concat(self, CGAffine other not None):
        self.real_transform = CGAffineTransformConcat(self.real_transform,
            other.real_transform)
        return self

    def __mul__(CGAffine x not None, CGAffine y not None):
        cdef CGAffineTransform new_transform
        new_transform = CGAffineTransformConcat(x.real_transform,
            y.real_transform)
        new_affine = CGAffine()
        set_affine_transform(new_affine, new_transform)
        return new_affine

    cdef void init_from_cgaffinetransform(self, CGAffineTransform t):
        self.real_transform = t

    def __div__(CGAffine x not None, CGAffine y not None):
        cdef CGAffineTransform new_transform
        new_transform = CGAffineTransformInvert(y.real_transform)
        new_affine = CGAffine()
        set_affine_transform(new_affine, CGAffineTransformConcat(x.real_transform, new_transform))
        return new_affine

    def apply_to_point(self, float x, float y):
        cdef CGPoint oldpoint
        oldpoint = CGPointMake(x, y)
        cdef CGPoint newpoint
        newpoint = CGPointApplyAffineTransform(oldpoint,
            self.real_transform)
        return newpoint.x, newpoint.y

    def apply_to_size(self, float width, float height):
        cdef CGSize oldsize
        oldsize = CGSizeMake(width, height)
        cdef CGSize newsize
        newsize = CGSizeApplyAffineTransform(oldsize, self.real_transform)
        return newsize.width, newsize.height

    def __repr__(self):
        return "CGAffine(%r, %r, %r, %r, %r, %r)" % (self.a, self.b, self.c,
            self.d, self.tx, self.ty)

    def as_matrix(self):
        return ((self.a, self.b, 0.0),
                (self.c, self.d, 0.0),
                (self.tx,self.ty,1.0))

cdef set_affine_transform(CGAffine t, CGAffineTransform newt):
    t.init_from_cgaffinetransform(newt)

##cdef class Point:
##    cdef CGPoint real_point
##
##    property x:
##        def __get__(self):
##            return self.real_point.x
##        def __set__(self, float value):
##            self.real_point.x = value
##
##    property y:
##        def __get__(self):
##            return self.real_point.y
##        def __set__(self, float value):
##            self.real_point.y = value
##
##    def __init__(self, float x, float y):
##        self.real_point = CGPointMake(x, y)
##
##    def apply_transform(self, CGAffine transform not None):
##        self.real_point = CGPointApplyTransform(self.real_point,
##            transform.real_transform)

cdef class Rect:
    cdef CGRect real_rect

    property x:
        def __get__(self):
            return self.real_rect.origin.x
        def __set__(self, float value):
            self.real_rect.origin.x = value

    property y:
        def __get__(self):
            return self.real_rect.origin.y
        def __set__(self, float value):
            self.real_rect.origin.y = value

    property width:
        def __get__(self):
            return self.real_rect.size.width
        def __set__(self, float value):
            self.real_rect.size.width = value

    property height:
        def __get__(self):
            return self.real_rect.size.height
        def __set__(self, float value):
            self.real_rect.size.height = value

    property min_x:
        def __get__(self):
            return CGRectGetMinX(self.real_rect)

    property max_x:
        def __get__(self):
            return CGRectGetMaxX(self.real_rect)

    property min_y:
        def __get__(self):
            return CGRectGetMinY(self.real_rect)

    property max_y:
        def __get__(self):
            return CGRectGetMaxY(self.real_rect)

    property mid_x:
        def __get__(self):
            return CGRectGetMidX(self.real_rect)

    property mid_y:
        def __get__(self):
            return CGRectGetMidY(self.real_rect)

    property is_null:
        def __get__(self):
            return CGRectIsNull(self.real_rect)

    property is_empty:
        def __get__(self):
            return CGRectIsEmpty(self. real_rect)

    def __init__(self, float x=0.0, float y=0.0, float width=0.0, float
        height=0.0):
        self.real_rect = CGRectMake(x,y,width,height)

    def intersects(self, Rect other not None):
        return CGRectIntersectsRect(self.real_rect, other.real_rect)

    def contains_rect(self, Rect other not None):
        return CGRectContainsRect(self.real_rect, other.real_rect)

    def contains_point(self, float x, float y):
        return CGRectContainsPoint(self.real_rect, CGPointMake(x,y))

    def __richcmp__(Rect x not None, Rect y not None, int op):
        if op == 2:
            return CGRectEqualToRect(x.real_rect, y.real_rect)
        elif op == 3:
            return not CGRectEqualToRect(x.real_rect, y.real_rect)
        else:
            raise NotImplementedError("only (in)equality can be tested")

    def standardize(self):
        self.real_rect = CGRectStandardize(self.real_rect)
        return self

    def inset(self, float x, float y):
        cdef CGRect new_rect
        new_rect = CGRectInset(self.real_rect, x, y)
        rect = Rect()
        set_rect(rect, new_rect)
        return rect

    def offset(self, float x, float y):
        cdef CGRect new_rect
        new_rect = CGRectOffset(self.real_rect, x, y)
        rect = Rect()
        set_rect(rect, new_rect)
        return rect

    def integral(self):
        self.real_rect = CGRectIntegral(self.real_rect)
        return self

    def __add__(Rect x not None, Rect y not None):
        cdef CGRect new_rect
        new_rect = CGRectUnion(x.real_rect, y.real_rect)
        rect = Rect()
        set_rect(rect, new_rect)
        return rect

    def union(self, Rect other not None):
        cdef CGRect new_rect
        new_rect = CGRectUnion(self.real_rect, other.real_rect)
        rect = Rect()
        set_rect(rect, new_rect)
        return rect

    def intersection(self, Rect other not None):
        cdef CGRect new_rect
        new_rect = CGRectIntersection(self.real_rect, other.real_rect)
        rect = Rect()
        set_rect(rect, new_rect)
        return rect

    def divide(self, float amount, edge):
        cdef CGRect slice
        cdef CGRect remainder
        CGRectDivide(self.real_rect, &slice, &remainder, amount, edge)
        pyslice = Rect()
        set_rect(pyslice, slice)
        pyrem = Rect()
        set_rect(pyrem, remainder)
        return pyslice, pyrem

    cdef init_from_cgrect(self, CGRect cgrect):
        self.real_rect = cgrect

    def __repr__(self):
        return "Rect(%r, %r, %r, %r)" % (self.x, self.y, self.width,
            self.height)

cdef set_rect(Rect pyrect, CGRect cgrect):
    pyrect.init_from_cgrect(cgrect)

cdef class CGMutablePath:
    cdef CGMutablePathRef path

    def __init__(self, CGMutablePath path=None):
        if path is not None:
            self.path = CGPathCreateMutableCopy(path.path)
        else:
            self.path = CGPathCreateMutable()

    def begin_path(self):
        return

    def move_to(self, float x, float y, CGAffine transform=None):
        cdef CGAffineTransform *ptr
        ptr = NULL
        if transform is not None:
            ptr = &(transform.real_transform)
        CGPathMoveToPoint(self.path, ptr, x, y)

    def arc(self, float x, float y, float r, float startAngle, float endAngle,
        bool clockwise=False, CGAffine transform=None):

        cdef CGAffineTransform *ptr
        ptr = NULL
        if transform is not None:
            ptr = &(transform.real_transform)

        CGPathAddArc(self.path, ptr, x, y, r, startAngle, endAngle, clockwise)

    def arc_to(self, float x1, float y1, float x2, float y2, float r,
        CGAffine transform=None):

        cdef CGAffineTransform *ptr
        ptr = NULL
        if transform is not None:
            ptr = &(transform.real_transform)

        CGPathAddArcToPoint(self.path, ptr, x1,y1, x2,y2, r)

    def curve_to(self, float cx1, float cy1, float cx2, float cy2, float x,
        float y, CGAffine transform=None):

        cdef CGAffineTransform *ptr
        ptr = NULL
        if transform is not None:
            ptr = &(transform.real_transform)

        CGPathAddCurveToPoint(self.path, ptr, cx1, cy1, cx2, cy2, x, y)

    def line_to(self, float x, float y, CGAffine transform=None):
        cdef CGAffineTransform *ptr
        ptr = NULL
        if transform is not None:
            ptr = &(transform.real_transform)

        CGPathAddLineToPoint(self.path, ptr, x, y)

    def lines(self, points, CGAffine transform=None):
        cdef CGAffineTransform *ptr
        ptr = NULL
        if transform is not None:
            ptr = &(transform.real_transform)

        cdef int n
        n = len(points)
        cdef int i

        CGPathMoveToPoint(self.path, ptr, points[0][0], points[0][1])

        for i from 1 <= i < n:
            CGPathAddLineToPoint(self.path, ptr, points[i][0], points[i][1])

    def add_path(self, CGMutablePath other_path not None, CGAffine transform=None):
        cdef CGAffineTransform *ptr
        ptr = NULL
        if transform is not None:
            ptr = &(transform.real_transform)

        CGPathAddPath(self.path, ptr, other_path.path)

    def quad_curve_to(self, float cx, float cy, float x, float y, CGAffine transform=None):
        cdef CGAffineTransform *ptr
        ptr = NULL
        if transform is not None:
            ptr = &(transform.real_transform)

        CGPathAddQuadCurveToPoint(self.path, ptr, cx, cy, x, y)

    def rect(self, float x, float y, float sx, float sy, CGAffine transform=None):
        cdef CGAffineTransform *ptr
        ptr = NULL
        if transform is not None:
            ptr = &(transform.real_transform)

        CGPathAddRect(self.path, ptr, CGRectMake(x,y,sx,sy))

    def rects(self, rects, CGAffine transform=None):
        cdef CGAffineTransform *ptr
        ptr = NULL
        if transform is not None:
            ptr = &(transform.real_transform)

        cdef int n
        n = len(rects)
        cdef int i
        for i from 0 <= i < n:
            CGPathAddRect(self.path, ptr, CGRectMakeFromPython(rects[i]))

    def close_path(self):
        CGPathCloseSubpath(self.path)

    def is_empty(self):
        return CGPathIsEmpty(self.path)

    def get_current_point(self):
        cdef CGPoint point
        point = CGPathGetCurrentPoint(self.path)
        return point.x, point.y

    def get_bounding_box(self):
        cdef CGRect rect
        rect = CGPathGetBoundingBox(self.path)
        return (rect.origin.x, rect.origin.y,
                rect.size.width, rect.size.height)

    def __richcmp__(CGMutablePath x not None, CGMutablePath y not None, int op):
        if op == 2:
            # testing for equality
            return CGPathEqualToPath(x.path, y.path)
        elif op == 3:
            # testing for inequality
            return not CGPathEqualToPath(x.path, y.path)
        else:
            raise NotImplementedError("only (in)equality tests are allowed")

    def __dealloc__(self):
        if self.path != NULL:
            CGPathRelease(self.path)
            self.path = NULL

cdef class _Markers:

    def get_marker(self, int marker_type, float size=1.0):
        """ Return the CGMutablePath corresponding to the given marker
        enumeration.

          Marker.get_marker(marker_type, size=1.0)

        Parameters
        ----------
        marker_type : int
            One of the enumerated marker types in kiva.constants.
        size : float, optional
            The linear size in points of the marker. Some markers (e.g. dot)
            ignore this.

        Returns
        -------
        path : CGMutablePath
        """

        if marker_type == constants.NO_MARKER:
            return CGMutablePath()
        elif marker_type == constants.SQUARE_MARKER:
            return self.square(size)
        elif marker_type == constants.DIAMOND_MARKER:
            return self.diamond(size)
        elif marker_type == constants.CIRCLE_MARKER:
            return self.circle(size)
        elif marker_type == constants.CROSSED_CIRCLE_MARKER:
            raise NotImplementedError
        elif marker_type == constants.CROSS_MARKER:
            return self.cross(size)
        elif marker_type == constants.TRIANGLE_MARKER:
            raise NotImplementedError
        elif marker_type == constants.INVERTED_TRIANGLE_MARKER:
            raise NotImplementedError
        elif marker_type == constants.PLUS_MARKER:
            raise NotImplementedError
        elif marker_type == constants.DOT_MARKER:
            raise NotImplementedError
        elif marker_type == constants.PIXEL_MARKER:
            raise NotImplementedError


    def square(self, float size):
        cdef float half
        half = size / 2

        m = CGMutablePath()
        m.rect(-half,-half,size,size)
        return m

    def diamond(self, float size):
        cdef float half
        half = size / 2

        m = CGMutablePath()
        m.move_to(0.0, -half)
        m.line_to(-half, 0.0)
        m.line_to(0.0, half)
        m.line_to(half, 0.0)
        m.close_path()
        return m

    def x(self, float size):
        cdef float half
        half = size / 2

        m = CGMutablePath()
        m.move_to(-half,-half)
        m.line_to(half,half)
        m.move_to(-half,half)
        m.line_to(half,-half)
        return m

    def cross(self, float size):
        cdef float half
        half = size / 2

        m = CGMutablePath()
        m.move_to(0.0, -half)
        m.line_to(0.0, half)
        m.move_to(-half, 0.0)
        m.line_to(half, 0.0)
        return m

    def dot(self):
        m = CGMutablePath()
        m.rect(-0.5,-0.5,1.0,1.0)
        return m

    def circle(self, float size):
        cdef float half
        half = size / 2
        m = CGMutablePath()
        m.arc(0.0, 0.0, half, 0.0, 6.2831853071795862, 1)
        return m

Markers = _Markers()

cdef class ShadingFunction:
    cdef CGFunctionRef function

    cdef void _setup_function(self, CGFunctionEvaluateCallback callback):
        cdef int i
        cdef CGFunctionCallbacks callbacks
        callbacks.version = 0
        callbacks.releaseInfo = NULL
        callbacks.evaluate = <CGFunctionEvaluateCallback>callback

        cdef CGFloat domain_bounds[2]
        cdef CGFloat range_bounds[8]

        domain_bounds[0] = 0.0
        domain_bounds[1] = 1.0
        for i from 0 <= i < 4:
            range_bounds[2*i] = 0.0
            range_bounds[2*i+1] = 1.0

        self.function = CGFunctionCreate(<void*>self, 1, domain_bounds,
            4, range_bounds, &callbacks)
        if self.function == NULL:
            raise RuntimeError("could not make CGFunctionRef")

cdef void shading_callback(object self, CGFloat* in_data, CGFloat* out_data):
    cdef int i
    out = self(in_data[0])
    for i from 0 <= i < self.n_dims:
        out_data[i] = out[i]

cdef class Shading:
    cdef CGShadingRef shading
    cdef public object function
    cdef int n_dims

    def __init__(self, ShadingFunction func not None):
        raise NotImplementedError("use AxialShading or RadialShading")

    def __dealloc__(self):
        if self.shading != NULL:
            CGShadingRelease(self.shading)

cdef class AxialShading(Shading):
    def __init__(self, ShadingFunction func not None, object start, object end,
        int extend_start=0, int extend_end=0):

        self.n_dims = 4

        cdef CGPoint start_point, end_point
        start_point = CGPointMake(start[0], start[1])
        end_point = CGPointMake(end[0], end[1])

        self.function = func

        cdef CGColorSpaceRef space
        space = CGColorSpaceCreateDeviceRGB()
        self.shading = CGShadingCreateAxial(space, start_point, end_point,
            func.function, extend_start, extend_end)
        CGColorSpaceRelease(space)
        if self.shading == NULL:
            raise RuntimeError("could not make CGShadingRef")

cdef class RadialShading(Shading):
    def __init__(self, ShadingFunction func not None, object start,
        float start_radius, object end, float end_radius, int extend_start=0,
        int extend_end=0):

        self.n_dims = 4

        cdef CGPoint start_point, end_point
        start_point = CGPointMake(start[0], start[1])
        end_point = CGPointMake(end[0], end[1])

        self.function = func

        cdef CGColorSpaceRef space
        space = CGColorSpaceCreateDeviceRGB()
        self.shading = CGShadingCreateRadial(space, start_point, start_radius,
            end_point, end_radius, func.function, extend_start, extend_end)
        CGColorSpaceRelease(space)
        if self.shading == NULL:
            raise RuntimeError("could not make CGShadingRef")

cdef void safe_free(void* mem):
    if mem != NULL:
        PyMem_Free(mem)

cdef class PiecewiseLinearColorFunction(ShadingFunction):
    cdef int num_stops
    cdef CGFloat* stops
    cdef CGFloat* red
    cdef CGFloat* green
    cdef CGFloat* blue
    cdef CGFloat* alpha

    def __init__(self, object stop_colors):
        cdef c_numpy.ndarray stop_array
        cdef int i

        stop_colors = numpy.array(stop_colors).astype(numpy.float32)

        if not (4 <= stop_colors.shape[0] <= 5) or len(stop_colors.shape) != 2:
            raise ValueError("need array [stops, red, green, blue[, alpha]]")

        if stop_colors[0,0] != 0.0 or stop_colors[0,-1] != 1.0:
            raise ValueError("stops need to start with 0.0 and end with 1.0")

        if not numpy.greater_equal(numpy.diff(stop_colors[0]), 0.0).all():
            raise ValueError("stops must be sorted and unique")

        self.num_stops = stop_colors.shape[1]
        self.stops = <CGFloat*>PyMem_Malloc(sizeof(CGFloat)*self.num_stops)
        self.red = <CGFloat*>PyMem_Malloc(sizeof(CGFloat)*self.num_stops)
        self.green = <CGFloat*>PyMem_Malloc(sizeof(CGFloat)*self.num_stops)
        self.blue = <CGFloat*>PyMem_Malloc(sizeof(CGFloat)*self.num_stops)
        self.alpha = <CGFloat*>PyMem_Malloc(sizeof(CGFloat)*self.num_stops)

        has_alpha = stop_colors.shape[0] == 5
        for i from 0 <= i < self.num_stops:
            self.stops[i] = stop_colors[0,i]
            self.red[i] = stop_colors[1,i]
            self.green[i] = stop_colors[2,i]
            self.blue[i] = stop_colors[3,i]
            if has_alpha:
                self.alpha[i] = stop_colors[4,i]
            else:
                self.alpha[i] = 1.0

        self._setup_function(piecewise_callback)

    def dump(self):
        cdef int i
        print('PiecewiseLinearColorFunction')
        print('  num_stops = %i' % self.num_stops)
        print('  stops = ', end=" ")
        for i from 0 <= i < self.num_stops:
            print(self.stops[i], end=" ")
        print()
        print('  red = ', end=" ")
        for i from 0 <= i < self.num_stops:
            print(self.red[i], end=" ")
        print()
        print('  green = ', end=" ")
        for i from 0 <= i < self.num_stops:
            print(self.green[i], end=" ")
        print()
        print('  blue = ', end=" ")
        for i from 0 <= i < self.num_stops:
            print(self.blue[i], end=" ")
        print()
        print('  alpha = ', end=" ")
        for i from 0 <= i < self.num_stops:
            print(self.alpha[i], end=" ")
        print()

    def __dealloc__(self):
        safe_free(self.stops)
        safe_free(self.red)
        safe_free(self.green)
        safe_free(self.blue)
        safe_free(self.alpha)


cdef int bisect_left(PiecewiseLinearColorFunction self, CGFloat t):
    cdef int lo, hi, mid
    cdef CGFloat stop

    hi = self.num_stops
    lo = 0
    while lo < hi:
        mid = (lo + hi)/2
        stop = self.stops[mid]
        if t < stop:
            hi = mid
        else:
            lo = mid + 1
    return lo

cdef void piecewise_callback(void* obj, CGFloat* t, CGFloat* out):
   cdef int i
   cdef CGFloat eps
   cdef PiecewiseLinearColorFunction self

   self = <PiecewiseLinearColorFunction>obj

   eps = 1e-6

   if fabs(t[0]) < eps:
       out[0] = self.red[0]
       out[1] = self.green[0]
       out[2] = self.blue[0]
       out[3] = self.alpha[0]
       return
   if fabs(t[0] - 1.0) < eps:
       i = self.num_stops - 1
       out[0] = self.red[i]
       out[1] = self.green[i]
       out[2] = self.blue[i]
       out[3] = self.alpha[i]
       return

   i = bisect_left(self, t[0])

   cdef CGFloat f, g, dx
   dx = self.stops[i] - self.stops[i-1]

   if dx > eps:
       f = (t[0]-self.stops[i-1])/dx
   else:
       f = 1.0

   g = 1.0 - f

   out[0] = f*self.red[i] + g*self.red[i-1]
   out[1] = f*self.green[i] + g*self.green[i-1]
   out[2] = f*self.blue[i] + g*self.blue[i-1]
   out[3] = f*self.alpha[i] + g*self.alpha[i-1]

cdef double _point_distance(double x1, double y1, double x2, double y2):
    cdef double dx = x1-x2
    cdef double dy = y1-y2
    return sqrt(dx*dx+dy*dy)

cdef bool _cgrect_within_circle(CGRect rect, double cx, double cy, double rad):
    cdef double d1 = _point_distance(cx,cy, rect.origin.x, rect.origin.y)
    cdef double d2 = _point_distance(cx,cy, rect.origin.x+rect.size.width, rect.origin.y)
    cdef double d3 = _point_distance(cx,cy, rect.origin.x+rect.size.width, rect.origin.y+rect.size.height)
    cdef double d4 = _point_distance(cx,cy, rect.origin.x, rect.origin.y+rect.size.height)
    return (d1<rad and d2<rad and d3<rad and d4<rad)

cdef bool _line_intersects_cgrect(double x, double y, double slope, CGRect rect):
    if slope == 0.:
        return rect.origin.x <= x <= (rect.origin.x+rect.size.width)
    elif isnan(slope):
        return rect.origin.y <= y <= (rect.origin.y+rect.size.height)
    # intersect the all sides
    cdef double left = rect.origin.x
    cdef double right = left + rect.size.width
    cdef double bottom = rect.origin.y
    cdef double top = bottom + rect.size.height
    if bottom <= (y + slope*(left-x)) <= top:
        return True
    if bottom <= (y + slope*(right-x)) <= top:
        return True
    if left <= (x + 1./slope*(top-y)) <= right:
        return True
    if left <= (x + 1./slope*(bottom-y)) <= right:
        return True
    return False


#### Font utilities ####

cdef CGColorRef _create_cg_color(object color):
    cdef CGFloat color_components[4]
    cdef CGColorRef cg_color

    color_components[0] = color[0]
    color_components[1] = color[1]
    color_components[2] = color[2]
    color_components[3] = color[3]

    cg_color = CGColorCreateGenericRGB(color_components[0], color_components[1],
                                       color_components[2], color_components[3])
    return cg_color

cdef CTLineRef _create_ct_line(object the_string, CTFontRef font, object stroke_color):
    cdef char* c_string
    cdef CFIndex text_len
    cdef CFStringRef cf_string
    cdef CFMutableAttributedStringRef cf_attr_string
    cdef CGColorRef cg_color
    cdef CTLineRef ct_line

    text_len = len(the_string)
    if text_len == 0:
        return NULL

    the_string = the_string.encode('utf-8')
    c_string = PyBytes_AsString(the_string)

    cf_string = CFStringCreateWithCString(NULL, c_string, kCFStringEncodingUTF8)
    cf_attr_string = CFAttributedStringCreateMutable(NULL, 0)
    CFAttributedStringReplaceString(cf_attr_string, CFRangeMake(0, 0), cf_string)
    CFRelease(cf_string)

    CFAttributedStringSetAttribute(cf_attr_string, CFRangeMake(0, text_len),
        kCTFontAttributeName, font)

    if stroke_color is not None:
        cg_color = _create_cg_color(stroke_color)
        CFAttributedStringSetAttribute(cf_attr_string, CFRangeMake(0, text_len),
            kCTForegroundColorAttributeName, cg_color)
        CGColorRelease(cg_color)

    # Stroke Color is supported by OS X 10.6 and greater using the
    # kCTStrokeColorAttributeName attribute.

    ct_line = CTLineCreateWithAttributedString(cf_attr_string)
    CFRelease(cf_attr_string)

    return ct_line
