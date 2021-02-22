# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
import numpy
import warnings
import wx

from enable.savage.svg.backends.null.null_renderer import NullRenderer


def _fixup_path_methods(path):
    def _new_add_rounded_rectangle(self, x, y, w, h, rx, ry):
        r = numpy.sqrt(rx * rx + ry * ry)
        self.AddRoundedRectangle(x, y, w, h, r)

    path.__class__.AddRoundedRectangleEx = _new_add_rounded_rectangle


class AbstractGradientBrush(object):
    """ Abstract base class for gradient brushes so they can be detected
    easily.
    """

    def IsOk(self):
        return True

    def bbox_transform(self, gc, bbox):
        """ Apply a transformation to make the bbox a unit square.
        """
        x0, y0, w, h = bbox
        gc.concat_ctm(((w, 0, 0), (0, h, 0), (x0, y0, 1)))


class Renderer(NullRenderer):

    NullBrush = wx.NullBrush
    NullGraphicsBrush = wx.NullGraphicsBrush
    NullPen = wx.NullPen
    TransparentPen = wx.TRANSPARENT_PEN

    caps = {
        "butt": wx.CAP_BUTT,
        "round": wx.CAP_ROUND,
        "square": wx.CAP_PROJECTING,
    }

    joins = {
        "miter": wx.JOIN_MITER,
        "round": wx.JOIN_ROUND,
        "bevel": wx.JOIN_BEVEL,
    }

    fill_rules = {"nonzero": wx.WINDING_RULE, "evenodd": wx.ODDEVEN_RULE}

    def __init__(self):
        pass

    @staticmethod
    def concatTransform(*args):
        return wx.GraphicsContext.ConcatTransform(*args)

    @staticmethod
    def createAffineMatrix(a, b, c, d, x, y):
        return wx.GraphicsRenderer.GetDefaultRenderer().CreateMatrix(
            a, b, c, d, x, y
        )

    @staticmethod
    def createBrush(color_tuple):
        return wx.Brush(wx.Colour(*color_tuple))

    @staticmethod
    def createNativePen(pen):
        return wx.GraphicsRenderer.GetDefaultRenderer().CreatePen(pen)

    @staticmethod
    def createPen(color_tuple):
        return wx.Pen(wx.Colour(*color_tuple))

    @staticmethod
    def createLinearGradientBrush(x1, y1, x2, y2, stops, spreadMethod="pad",
                                  transforms=None, units="userSpaceOnUse"):

        stops = numpy.transpose(stops)

        def convert_stop(stop):
            offset, red, green, blue, opacity = stop
            color = wx.Colour(
                red * 255, green * 255, blue * 255, opacity * 255
            )
            return offset, color

        if wx.VERSION[:2] > (2, 9):
            # wxPython 2.9+ supports a collection of stops
            wx_stops = wx.GraphicsGradientStops()
            for stop in stops:
                offset, color = convert_stop(stop)
                wx_stops.Add(color, offset)

            wx_renderer = wx.GraphicsRenderer.GetDefaultRenderer()
            return wx_renderer.CreateLinearGradientBrush(
                x1, y1, x2, y2, wx_stops
            )

        else:
            if len(stops) > 2:
                msg = ("wxPython 2.8 only supports 2 gradient stops, but %d "
                       "were specified")
                warnings.warn(msg % len(stops))

            start_offset, start_color = convert_stop(stops[0])
            end_offset, end_color = convert_stop(stops[1])

            wx_renderer = wx.GraphicsRenderer.GetDefaultRenderer()
            return wx_renderer.CreateLinearGradientBrush(
                x1, y1, x2, y2, start_color, end_color
            )

    @staticmethod
    def createRadialGradientBrush(cx, cy, r, stops, fx=None, fy=None,
                                  spreadMethod="pad", transforms=None,
                                  units="userSpaceOnUse"):

        stops = numpy.transpose(stops)

        def convert_stop(stop):
            offset, red, green, blue, opacity = stop
            color = wx.Colour(
                red * 255, green * 255, blue * 255, opacity * 255
            )
            return offset, color

        if wx.VERSION[:2] > (2, 9):
            # wxPython 2.9+ supports a collection of stops
            wx_stops = wx.GraphicsGradientStops()
            for stop in stops:
                offset, color = convert_stop(stop)
                wx_stops.Add(color, offset)

            wx_renderer = wx.GraphicsRenderer.GetDefaultRenderer()
            return wx_renderer.CreateRadialGradientBrush(
                fx, fy, cx, cy, r, wx_stops
            )

        else:

            if len(stops) > 2:
                msg = ("wxPython 2.8 only supports 2 gradient stops, but %d "
                       "were specified")
                warnings.warn(msg % len(stops))

            start_offset, start_color = convert_stop(stops[0])
            end_offset, end_color = convert_stop(stops[-1])

            if fx is None:
                fx = cx
            if fy is None:
                fy = cy

            wx_renderer = wx.GraphicsRenderer.GetDefaultRenderer()
            return wx_renderer.CreateRadialGradientBrush(
                fx, fy, cx, cy, r, start_color, end_color
            )

    @staticmethod
    def fillPath(*args):
        return wx.GraphicsContext.FillPath(*args)

    @staticmethod
    def getCurrentPoint(path):
        return path.GetCurrentPoint().Get()

    @staticmethod
    def getFont(font_name=wx.SYS_DEFAULT_GUI_FONT):
        return wx.SystemSettings.GetFont(font_name)

    @staticmethod
    def makeMatrix(*args):
        return wx.GraphicsRenderer.GetDefaultRenderer().CreateMatrix(*args)

    @staticmethod
    def makePath():
        path = wx.GraphicsRenderer.GetDefaultRenderer().CreatePath()
        _fixup_path_methods(path)
        return path

    @staticmethod
    def popState(*args):
        return wx.GraphicsContext.PopState(*args)

    @staticmethod
    def pushState(state):
        return wx.GraphicsContext.PushState(state)

    @staticmethod
    def rotate(dc, angle):
        return dc.Rotate(angle)

    @staticmethod
    def scale(*args):
        return wx.GraphicsContext.Scale(*args)

    @staticmethod
    def setBrush(*args):
        wx.GraphicsContext.SetBrush(*args)

    @staticmethod
    def setFontSize(font, size):
        if "__WXMSW__" in wx.PlatformInfo:
            i = int(size)
            font.SetPixelSize((i, i))
        else:
            font.SetPointSize(int(size))
        return font

    @classmethod
    def setFontStyle(cls, font, style):
        font.style = style

    @classmethod
    def setFontWeight(cls, font, weight):
        font.weight = weight

    @staticmethod
    def setPen(*args):
        wx.GraphicsContext.SetPen(*args)

    @staticmethod
    def setPenDash(pen, dasharray, offset):
        pen.SetDashes(dasharray)

    @staticmethod
    def setFont(context, font, brush):
        return context.SetFont(font, brush.Colour)

    @staticmethod
    def strokePath(*args):
        return wx.GraphicsContext.StrokePath(*args)

    @staticmethod
    def clipPath(gc, path):
        rect = path.GetBox()
        region = wx.Region(rect.x, rect.y, rect.width, rect.height)
        gc.ClipRegion(region)

    @staticmethod
    def translate(*args):
        return wx.GraphicsContext.Translate(*args)

    @staticmethod
    def DrawText(context, text, x, y, brush=NullGraphicsBrush, anchor="start"):
        # SVG spec appears to originate text from the bottom
        # rather than the top as with our API. This function
        # will measure and then re-orient the text as needed.
        w, h = context.GetTextExtent(text)

        if anchor != "start":
            if anchor == "middle":
                x -= w / 2.0
            elif anchor == "end":
                x -= w

        y -= h
        context.DrawText(text, x, y)

    @staticmethod
    def DrawImage(context, image, x, y, width, height):
        # ignore the width & height provided
        width = image.shape[1]
        height = image.shape[0]

        if image.shape[2] == 3:
            bmp = wx.BitmapFromBuffer(width, height, image.flatten())
        else:
            bmp = wx.BitmapFromBufferRGBA(width, height, image.flatten())

        context.DrawBitmap(bmp, x, y, width, height)
