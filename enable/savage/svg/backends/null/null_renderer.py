# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
import sys


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
        if sys.platform == "darwin":
            gc.concat_ctm(((w, 0, 0), (0, h, 0), (x0, y0, 1)))
        else:
            gc.concat_ctm((w, 0, 0, h, x0, y0))


class NullRenderer(object):
    NullBrush = None
    NullGraphicsBrush = None
    NullPen = None
    TransparentPen = None

    caps = {"butt": None, "round": None, "square": None}

    joins = {"miter": None, "round": None, "bevel": None}

    fill_rules = {"nonzero": None, "evenodd": None}

    def __init__(self):
        pass

    @classmethod
    def concatTransform(cls, gc, matrix):
        raise NotImplementedError()

    @classmethod
    def createAffineMatrix(cls, a, b, c, d, x, y):
        raise NotImplementedError()

    @classmethod
    def createBrush(cls, color_tuple):
        raise NotImplementedError()

    @classmethod
    def createNativePen(cls, pen):
        raise NotImplementedError()

    @classmethod
    def createPen(cls, color_tuple):
        raise NotImplementedError()

    @classmethod
    def createLinearGradientBrush(cls, x1, y1, x2, y2, stops,
                                  spreadMethod="pad", transforms=None,
                                  units="userSpaceOnUse"):
        raise NotImplementedError()

    @classmethod
    def createRadialGradientBrush(cls, cx, cy, r, stops, fx=None, fy=None,
                                  spreadMethod="pad", transforms=None,
                                  units="userSpaceOnUse"):
        raise NotImplementedError()

    @classmethod
    def getFont(cls, font_name="Arial"):
        raise NotImplementedError()

    @classmethod
    def makeMatrix(cls, *args):
        raise NotImplementedError()

    @classmethod
    def makePath(cls):
        raise NotImplementedError()

    @classmethod
    def popState(cls, gc):
        raise NotImplementedError()

    @classmethod
    def pushState(cls, gc):
        raise NotImplementedError()

    @classmethod
    def setFontSize(cls, font, size):
        raise NotImplementedError()

    @classmethod
    def setFontStyle(cls, font, style):
        raise NotImplementedError()

    @classmethod
    def setFontWeight(cls, font, weight):
        raise NotImplementedError()

    @classmethod
    def setFont(cls, gc, font, brush):
        raise NotImplementedError()

    @classmethod
    def setBrush(cls, gc, brush):
        raise NotImplementedError()

    @classmethod
    def setPenDash(cls, pen, dasharray, offset):
        raise NotImplementedError()

    @classmethod
    def setPen(cls, gc, pen):
        raise NotImplementedError()

    @classmethod
    def strokePath(cls, gc, path):
        raise NotImplementedError()

    @classmethod
    def fillPath(cls, gc, path, mode):
        raise NotImplementedError()

    @classmethod
    def gradientPath(cls, gc, path, brush):
        raise NotImplementedError()

    @classmethod
    def clipPath(cls, gc, path):
        raise NotImplementedError()

    @classmethod
    def translate(cls, gc, *args):
        raise NotImplementedError()

    @classmethod
    def rotate(cls, gc, angle):
        raise NotImplementedError()

    @classmethod
    def scale(cls, gc, sx, sy):
        raise NotImplementedError()

    @classmethod
    def GetTextExtent(cls, gc, text):
        raise NotImplementedError()

    @classmethod
    def DrawText(cls, gc, text, x, y, brush, anchor="start"):
        """ Draw text at the given x,y position with the color of the
            given brush.
        """
        raise NotImplementedError()

    @classmethod
    def DrawImage(cls, gc, image, x, y, width, height):
        raise NotImplementedError()
