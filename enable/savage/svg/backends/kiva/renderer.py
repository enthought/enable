# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
from math import pi
import sys
import warnings

import numpy as np

from enable.compiled_path import CompiledPath as KivaCompiledPath
from kiva import affine, constants, fonttools
from kiva.api import Font

from enable.savage.svg import svg_extras
from enable.savage.svg.backends.null.null_renderer import (
    NullRenderer,
    AbstractGradientBrush,
)

# Get the Canvas class for drawing on...


def _GetCurrentPoint(gc):
    total_vertices = gc.total_vertices()
    if total_vertices == 0:
        return (0.0, 0.0)
    return gc.vertex(total_vertices - 1)[0]


class CompiledPath(KivaCompiledPath):

    AddPath = KivaCompiledPath.add_path
    AddRectangle = KivaCompiledPath.rect
    MoveToPoint = KivaCompiledPath.move_to
    AddLineToPoint = KivaCompiledPath.line_to
    CloseSubpath = KivaCompiledPath.close_path
    if hasattr(KivaCompiledPath, "get_current_point"):
        GetCurrentPoint = KivaCompiledPath.get_current_point
    else:
        GetCurrentPoint = _GetCurrentPoint
    AddQuadCurveToPoint = KivaCompiledPath.quad_curve_to

    def AddCurveToPoint(self, ctrl1, ctrl2, endpoint):
        self.curve_to(
            ctrl1[0], ctrl1[1], ctrl2[0], ctrl2[1], endpoint[0], endpoint[1]
        )

    def AddEllipticalArcTo(self, x, y, w, h, theta0, dtheta, phi=0):
        arc = svg_extras.bezier_arc(x, y, x + w, y + h, theta0, dtheta)
        for i, (x1, y1, x2, y2, x3, y3, x4, y4) in enumerate(arc):
            self.curve_to(x2, y2, x3, y3, x4, y4)

    def elliptical_arc_to(self, rx, ry, phi, large_arc_flag, sweep_flag, x2,
                          y2):
        x1, y1 = self.GetCurrentPoint()
        arcs = svg_extras.elliptical_arc_to(
            self, rx, ry, phi, large_arc_flag, sweep_flag, x1, y1, x2, y2
        )

        for arc in arcs:
            self.curve_to(*arc)

    def AddCircle(self, x, y, r):
        self.arc(x, y, r, 0.0, 2 * pi)

    def AddEllipse(self, cx, cy, rx, ry):
        arc = svg_extras.bezier_arc(cx - rx, cy - ry, cx + rx, cy + ry, 0, 360)
        for i, (x1, y1, x2, y2, x3, y3, x4, y4) in enumerate(arc):
            if i == 0:
                self.move_to(x1, y1)
            self.curve_to(x2, y2, x3, y3, x4, y4)

    def AddRoundedRectangleEx(self, x, y, w, h, rx, ry):
        # origin
        self.move_to(x + rx, y)
        self.line_to(x + w - rx, y)
        # top right
        cx = rx * 2
        cy = ry * 2
        self.AddEllipticalArcTo(x + w - cx, y, cx, cy, 270, 90)
        self.AddLineToPoint(x + w, y + h - ry)
        # top left
        self.AddEllipticalArcTo(x + w - cx, y + h - cy, cx, cy, 0, 90)
        self.line_to(x + rx, y + h)
        # bottom left
        self.AddEllipticalArcTo(x, y + h - cy, cx, cy, 90, 90)
        self.line_to(x, y + ry)
        # bottom right
        self.AddEllipticalArcTo(x, y, cx, cy, 180, 90)
        self.close_path()


class Pen(object):
    def __init__(self, color):
        # fixme: what format is the color passed in? int or float
        self.color = color
        self.cap = constants.CAP_BUTT
        self.join = constants.JOIN_MITER
        self.width = 1
        self.dasharray = None
        self.dashoffset = 0.0

    def SetCap(self, cap):
        self.cap = cap

    def SetJoin(self, join):
        self.join = join

    def SetWidth(self, width):
        self.width = width

    def SetDash(self, dasharray, dashoffset=0.0):
        self.dasharray = dasharray
        self.dashoffset = dashoffset

    def set_on_gc(self, gc):
        """ Set the appropriate properties on the GraphicsContext.
        """
        # fixme: Should the pen affect text as well?
        # fixme: How about line style, thickness, etc.
        # translate from 0-255 to 0-1 values.
        color = tuple([x / 255.0 for x in self.color])
        gc.set_stroke_color(color)
        gc.set_line_join(self.join)
        gc.set_line_cap(self.cap)
        gc.set_line_width(self.width)
        if self.dasharray is not None:
            gc.set_line_dash(self.dasharray, self.dashoffset)


class ColorBrush(object):
    def __init__(self, color):
        # fixme: what format is the color passed in? int or float
        self.color = color
        # fixme: This was needed for a font setting in document.
        #        Fix this and remove.
        self.Colour = self.color

    def __repr__(self):
        return "ColorBrush(%r)" % (self.color,)

    def IsOk(self):
        return True

    def set_on_gc(self, gc):
        """ Set the appropriate properties on the GraphicsContext.
        """
        # translate from 0-255 to 0-1 values.
        try:
            color = tuple([x / 255.0 for x in list(self.color)])
        except Exception:
            color = (0, 0, 0, 1)
        gc.set_fill_color(color)


class LinearGradientBrush(AbstractGradientBrush):
    """ A Brush representing a linear gradient.
    """

    def __init__(self, x1, y1, x2, y2, stops, spreadMethod="pad",
                 transforms=None, units="userSpaceOnUse"):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.stops = stops
        self.spreadMethod = spreadMethod
        if transforms is None:
            transforms = []
        self.transforms = transforms
        self.units = units

    def __repr__(self):
        return (
            "LinearGradientBrush(%r,%r, %r,%r, %r, spreadMethod=%r, "
            "transforms=%r, units=%r)"
            % (
                self.x1,
                self.y1,
                self.x2,
                self.y2,
                self.stops,
                self.spreadMethod,
                self.transforms,
                self.units,
            )
        )

    def set_on_gc(self, gc, bbox=None):

        # Apply transforms
        if self.transforms is not None:
            for func, f_args in self.transforms:
                if isinstance(f_args, tuple):
                    func(gc, *f_args)
                else:
                    func(gc, f_args)

        x1 = self.x1
        x2 = self.x2
        y1 = self.y1
        y2 = self.y2

        if sys.platform == "darwin":
            if self.spreadMethod != "pad":
                warnings.warn(
                    "spreadMethod %r is not supported. Using 'pad'"
                    % self.spreadMethod
                )

            if bbox is not None:
                gc.clip_to_rect(*bbox)

        else:
            if self.units == "objectBoundingBox" and bbox is not None:
                x1 = (bbox[2] + bbox[0]) * x1
                y1 = (bbox[3] + bbox[1]) * y1
                x2 = (bbox[2] + bbox[0]) * x2
                y2 = (bbox[3] + bbox[1]) * y2

                self.bbox_transform(gc, bbox)

        stops = np.transpose(self.stops)
        gc.linear_gradient(
            x1, y1, x2, y2, stops, self.spreadMethod, self.units
        )


class RadialGradientBrush(AbstractGradientBrush):
    """ A Brush representing a radial gradient.
    """

    def __init__(self, cx, cy, r, stops, fx=None, fy=None, spreadMethod="pad",
                 transforms=None, units="userSpaceOnUse"):
        self.cx = cx
        self.cy = cy
        self.r = r
        self.stops = stops
        if fx is None:
            fx = self.cx
        self.fx = fx
        if fy is None:
            fy = self.cy
        self.fy = fy
        self.spreadMethod = spreadMethod
        if transforms is None:
            transforms = []
        self.transforms = transforms
        self.units = units

    def __repr__(self):
        return (
            "RadialGradientBrush(%r,%r, %r, %r, fx=%r,fy=%r, "
            "spreadMethod=%r, transforms=%r, units=%r)"
            % (
                self.cx,
                self.cy,
                self.r,
                self.stops,
                self.fx,
                self.fy,
                self.spreadMethod,
                self.transforms,
                self.units,
            )
        )

    def set_on_gc(self, gc, bbox=None):

        if self.transforms is not None:
            for func, f_args in self.transforms:
                if isinstance(f_args, tuple):
                    func(gc, *f_args)
                else:
                    func(gc, f_args)

        cx = self.cx
        cy = self.cy
        r = self.r
        fx = self.fx
        fy = self.fy

        if sys.platform == "darwin":
            if self.spreadMethod != "pad":
                warnings.warn(
                    "spreadMethod %r is not supported. Using 'pad'"
                    % self.spreadMethod
                )

            if bbox is not None:
                gc.clip_to_rect(*bbox)

        else:
            if self.units == "objectBoundingBox" and bbox is not None:
                cx = (bbox[2] + bbox[0]) * cx
                cy = (bbox[3] + bbox[1]) * cy
                fx = (bbox[2] + bbox[0]) * fx
                fy = (bbox[3] + bbox[1]) * fy
                r *= np.sqrt(
                    (bbox[2] - bbox[0]) ** 2 + (bbox[3] - bbox[1]) ** 2
                )

                self.bbox_transform(gc, bbox)

        stops = np.transpose(self.stops)
        gc.radial_gradient(
            cx, cy, r, fx, fy, stops, self.spreadMethod, self.units
        )


def font_style(font):
    """ Return a string for the font style of a given font.

        fixme: Shouldn't the backends handle this?
    """

    if font.style == "italic" and font.weight == "bold":
        style = "bold italic"
    elif font.style == "italic":
        style = "italic"
    elif font.weight == "bold":
        style = "bold"
    elif font.style in [0, "regular", "normal"]:
        style = "regular"
    else:
        print(
            "Font style '%s' and weight: '%s' not known."
            " Using style='regular'" % (font.style, font.weight)
        )
        style = "regular"

    return style


class Renderer(NullRenderer):
    # fimxe: Shouldn't this just be the GraphicsContext?

    NullBrush = None

    NullGraphicsBrush = None
    NullPen = None
    TransparentPen = Pen((1.0, 1.0, 1.0, 0.0))

    caps = {
        "butt": constants.CAP_BUTT,
        "round": constants.CAP_ROUND,
        "square": constants.CAP_SQUARE,
    }

    joins = {
        "miter": constants.JOIN_MITER,
        "round": constants.JOIN_ROUND,
        "bevel": constants.JOIN_BEVEL,
    }

    fill_rules = {"nonzero": constants.FILL, "evenodd": constants.EOF_FILL}

    def __init__(self):
        pass

    @classmethod
    def concatTransform(cls, gc, matrix):
        return gc.concat_ctm(matrix)

    @classmethod
    def createAffineMatrix(cls, a, b, c, d, x, y):
        # FIXME: should we create a 6x1 or 3x3 matrix???
        return (a, b, c, d, x, y)

    @classmethod
    def createBrush(cls, color_tuple):
        return ColorBrush(color_tuple)

    @classmethod
    def createNativePen(cls, pen):
        # fixme: Not really sure what to do here...
        return pen

    @classmethod
    def createPen(cls, color_tuple):
        return Pen(color_tuple)

    @classmethod
    def createLinearGradientBrush(cls, x1, y1, x2, y2, stops,
                                  spreadMethod="pad", transforms=None,
                                  units="userSpaceOnUse"):
        return LinearGradientBrush(
            x1, y1, x2, y2, stops, spreadMethod, transforms, units
        )

    @classmethod
    def createRadialGradientBrush(cls, cx, cy, r, stops, fx=None, fy=None,
                                  spreadMethod="pad", transforms=None,
                                  units="userSpaceOnUse"):
        return RadialGradientBrush(
            cx, cy, r, stops, fx, fy, spreadMethod, transforms, units
        )

    @classmethod
    def getCurrentPoint(cls, path):
        return path.GetCurrentPoint()

    @classmethod
    def getFont(cls, font_name="Arial"):
        kiva_style = constants.NORMAL
        if "-" in font_name:
            font_name, style = font_name.split("-", 2)
            style = style.lower()
            if "bold" in style:
                kiva_style += constants.BOLD
            if "italic" in style:
                kiva_style += constants.ITALIC
        return Font(font_name, style=kiva_style)

    @classmethod
    def makeMatrix(cls, *args):
        raise NotImplementedError()

    @classmethod
    def makePath(cls):
        return CompiledPath()

    @classmethod
    def popState(cls, gc):
        return gc.restore_state()

    @classmethod
    def pushState(cls, gc):
        return gc.save_state()

    @classmethod
    def setFontSize(cls, font, size):
        # Agg expects only integer fonts
        font.size = int(size)
        return font

    @classmethod
    def setFontStyle(cls, font, style):
        if isinstance(style, str):
            if style not in fonttools.font.font_styles:
                warnings.warn('font style "%s" not supported' % style)
            else:
                font.style = fonttools.font.font_styles[style]
        else:
            font.style = style

    @classmethod
    def setFontWeight(cls, font, weight):
        if isinstance(weight, str):
            if weight not in fonttools.font.font_weights:
                warnings.warn('font weight "%s" not supported' % weight)
            else:
                font.weight = fonttools.font.font_weights[weight]
        else:
            font.weight = weight

    @classmethod
    def setFont(cls, gc, font, brush):
        color = tuple([c / 255.0 for c in getattr(brush, "color", (0, 0, 0))])

        # text color is controlled by stroke instead of fill color in kiva.
        gc.set_stroke_color(color)

        try:
            # fixme: The Mac backend doesn't accept style/width as non-integers
            #        in set_font, but does for select_font...
            if sys.platform == "darwin":
                style = font_style(font)
                gc.select_font(font.face_name, font.size, style=style)
            else:
                gc.set_font(font)

        except ValueError:
            warnings.warn(
                "failed to find set '%s'.  Using Arial" % font.face_name
            )
            if sys.platform == "darwin":
                style = font_style(font)
                gc.select_font("Arial", font.size, style)
            else:
                gc.set_font(font)

    @classmethod
    def setBrush(cls, gc, brush):
        if brush is Renderer.NullBrush:
            # fixme: What do I do in this case?  Seem
            pass
        else:
            brush.set_on_gc(gc)

    @classmethod
    def setPen(cls, gc, pen):
        pen.set_on_gc(gc)

    @classmethod
    def setPenDash(cls, pen, dasharray, offset):
        pen.SetDash(dasharray, offset)

    @classmethod
    def strokePath(cls, gc, path):
        # fixme: Do we need to clear the path first?
        gc.add_path(path)
        return gc.stroke_path()

    @classmethod
    def fillPath(cls, gc, path, mode):
        # fixme: Do we need to clear the path first?
        gc.add_path(path)
        return gc.draw_path(mode)

    @classmethod
    def gradientPath(cls, gc, path, brush):
        gc.save_state()
        gc.add_path(path)

        gc.clip()
        if hasattr(path, "get_bounding_box"):
            bbox = path.get_bounding_box()
        else:
            bbox = [np.inf, np.inf, -np.inf, -np.inf]
            for i in range(path.total_vertices()):
                vertex = path.vertex(i)
                bbox[0] = min(bbox[0], vertex[0][0])
                bbox[1] = min(bbox[1], vertex[0][1])
                bbox[2] = max(bbox[2], vertex[0][0])
                bbox[3] = max(bbox[3], vertex[0][1])

        brush.set_on_gc(gc, bbox=bbox)

        gc.close_path()
        gc.fill_path()
        gc.restore_state()

    @classmethod
    def clipPath(cls, gc, path):
        gc.add_path(path)
        return gc.clip()

    @classmethod
    def translate(cls, gc, *args):
        return gc.translate_ctm(*args)

    @classmethod
    def rotate(cls, gc, angle):
        return gc.rotate_ctm(angle)

    @classmethod
    def scale(cls, gc, sx, sy):
        return gc.scale_ctm(sx, sy)

    @classmethod
    def GetTextExtent(cls, gc, text):
        x, y, w, h = gc.get_text_extent(text)
        return w, h

    @classmethod
    def DrawText(cls, gc, text, x, y, brush, anchor="start"):
        """ Draw text at the given x,y position with the color of the
            given brush.

            fixme: Handle gradients...?
        """
        gc.save_state()
        try:
            color = tuple(
                [c / 255.0 for c in getattr(brush, "color", (0, 0, 0))]
            )
            # Setting stroke instead of fill color because that is
            # what kiva uses.
            gc.set_stroke_color(color)

            # PDF (our API) has the origin in the lower left.
            # SVG (what we are rendering) has the origin in the upper right.
            # The ctm (our API) has been set up with a scaling and translation
            # to draw as if the upper right is the origin and positive is down
            # so that the SVG will render correctly.  This works great accept
            # for text which will render up side down.  To fix this, we set the
            # text transform matrix to have y going up so the text is rendered
            # upright.  But since, +y is now *up*, we need to draw at -y.

            if anchor != "start":
                tx, ty, tw, th = gc.get_text_extent(text)
                if anchor == "middle":
                    x -= tw / 2.0
                elif anchor == "end":
                    x -= tw
            gc.scale_ctm(1.0, -1.0)
            gc.show_text_at_point(text, x, -y)
        finally:
            gc.restore_state()

    @classmethod
    def DrawImage(cls, gc, image, x, y, width, height):
        gc.save_state()
        gc.translate_ctm(x, y + height)
        gc.scale_ctm(1.0, -1.0)
        gc.draw_image(image, (0, 0, width, height))
        gc.restore_state()
