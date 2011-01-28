import sys

class AbstractGradientBrush(object):
    """ Abstract base class for gradient brushes so they can be detected easily.
    """

    def IsOk(self):
        return True

    def bbox_transform(self, gc, bbox):
        """ Apply a transformation to make the bbox a unit square.
        """
        x0, y0, w, h = bbox
        if sys.platform == 'darwin':
            gc.concat_ctm(((w, 0, 0), (0, h, 0), (x0, y0, 1)))
        else:
            gc.concat_ctm((w,0,0,h,x0,y0))


class NullRenderer(object):
    NullBrush         = None
    NullGraphicsBrush = None
    NullPen           = None
    TransparentPen    = None

    caps = {
            'butt':None,
            'round':None,
            'square':None
            }

    joins = {
            'miter':None,
            'round':None,
            'bevel':None
            }

    fill_rules = {'nonzero':None, 'evenodd': None}

    def __init__(self):
        pass

    @classmethod
    def concatTransform(cls, gc, matrix):
        raise NotImplemented()

    @classmethod
    def createAffineMatrix(cls, a,b,c,d,x,y):
        raise NotImplemented()

    @classmethod
    def createBrush(cls, color_tuple):
        raise NotImplemented()

    @classmethod
    def createNativePen(cls, pen):
        raise NotImplemented()

    @classmethod
    def createPen(cls, color_tuple):
        raise NotImplemented()

    @classmethod
    def createLinearGradientBrush(cls, x1,y1,x2,y2, stops, spreadMethod='pad',
                                  transforms=None, units='userSpaceOnUse'):
        raise NotImplemented()

    @classmethod
    def createRadialGradientBrush(cls, cx,cy, r, stops, fx=None,fy=None,
                                  spreadMethod='pad', transforms=None,
                                  units='userSpaceOnUse'):
        raise NotImplemented()

    @classmethod
    def getFont(cls, font_name='Arial'):
        raise NotImplemented()

    @classmethod
    def makeMatrix(cls, *args):
        raise NotImplemented()

    @classmethod
    def makePath(cls):
        raise NotImplemented()

    @classmethod
    def popState(cls, gc):
        raise NotImplemented()

    @classmethod
    def pushState(cls, gc):
        raise NotImplemented()

    @classmethod
    def setFontSize(cls, font, size):
        raise NotImplemented()

    @classmethod
    def setFontStyle(cls, font, style):
        raise NotImplemented()

    @classmethod
    def setFontWeight(cls, font, weight):
        raise NotImplemented()

    @classmethod
    def setFont(cls, gc, font, brush):
        raise NotImplemented()

    @classmethod
    def setBrush(cls, gc, brush):
        raise NotImplemented()

    @classmethod
    def setPenDash(cls, pen, dasharray, offset):
        raise NotImplemented()

    @classmethod
    def setPen(cls, gc, pen):
        raise NotImplemented()

    @classmethod
    def strokePath(cls, gc, path):
        raise NotImplemented()

    @classmethod
    def fillPath(cls, gc, path, mode):
        raise NotImplemented()

    @classmethod
    def gradientPath(cls, gc, path, brush):
        raise NotImplemented()

    @classmethod
    def clipPath(cls, gc, path):
        raise NotImplemented()

    @classmethod
    def translate(cls, gc, *args):
        raise NotImplemented()

    @classmethod
    def rotate(cls, gc, angle):
        raise NotImplemented()

    @classmethod
    def scale(cls, gc, sx, sy):
        raise NotImplemented()

    @classmethod
    def GetTextExtent(cls, gc, text):
        raise NotImplemented()

    @classmethod
    def DrawText(cls, gc, text, x, y, brush, anchor='start'):
        """ Draw text at the given x,y position with the color of the
            given brush.
        """
        raise NotImplemented()

    @classmethod
    def DrawImage(cls, gc, image, x, y, width, height):
        raise NotImplemented()
