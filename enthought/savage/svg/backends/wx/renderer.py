import wx

from enthought.savage.svg.backends.null.null_renderer import NullRenderer

class AbstractGradientBrush(object):
    """ Abstract base class for gradient brushes so they can be detected easily.
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
            'butt':wx.CAP_BUTT,
            'round':wx.CAP_ROUND,
            'square':wx.CAP_PROJECTING
            }
    
    joins = {
            'miter':wx.JOIN_MITER,
            'round':wx.JOIN_ROUND,
            'bevel':wx.JOIN_BEVEL
            }
    
    fill_rules = {'nonzero':wx.WINDING_RULE, 'evenodd': wx.ODDEVEN_RULE}
    
    def __init__(self):
        pass

    @staticmethod
    def concatTransform(*args):
        return wx.GraphicsContext.ConcatTransform(*args)

    @staticmethod
    def createAffineMatrix(a,b,c,d,x,y):
        return wx.GraphicsRenderer_GetDefaultRenderer().CreateMatrix(a,b,c,d,x,y)

    @staticmethod
    def createBrush(color_tuple):
        return wx.Brush(wx.Colour(*color_tuple))
    
    @staticmethod
    def createNativePen(pen):
        return wx.GraphicsRenderer_GetDefaultRenderer().CreatePen(pen)
    
    @staticmethod
    def createPen(color_tuple):
        return wx.Pen(wx.Colour(*color_tuple))
    
    @staticmethod
    def createLinearGradientBrush(x1,y1,x2,y2, start_color_tuple, end_color_tuple):
        start_color = wx.Colour(*start_color_tuple)
        end_color = wx.Colour(*end_color_tuple)
        wx_renderer = wx.GraphicsRenderer.GetDefaultRenderer()
        return wx_renderer.CreateLinearGradientBrush(x1, y1, x2, y2, 
                                                     start_color, end_color)

    @staticmethod
    def createRadialGradientBrush(xo, yo, xc, yc, radius, start_color_tuple, end_color_tuple):
        start_color = wx.Colour(*start_color_tuple)
        end_color = wx.Colour(*end_color_tuple)
        wx_renderer = wx.GraphicsRenderer.GetDefaultRenderer()
        return wx_renderer.CreateRadialGradientBrush(xo, yo, xc, yc, radius, 
                                                     start_color, end_color)

    @staticmethod
    def fillPath(*args):
        return wx.GraphicsContext.FillPath(*args)

    @staticmethod
    def getFont(font_name=wx.SYS_DEFAULT_GUI_FONT):
        return wx.SystemSettings.GetFont(font_name)

    @staticmethod
    def makeMatrix(*args):
        return wx.GraphicsRenderer_GetDefaultRenderer().CreateMatrix(*args)
    
    @staticmethod
    def makePath():
        return wx.GraphicsRenderer_GetDefaultRenderer().CreatePath()        

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
        if '__WXMSW__' in wx.PlatformInfo:
            i = int(size)
            font.SetPixelSize((i, i))
        else:
            font.SetPointSize(int(size))
        return font
        

    @staticmethod
    def setPen(*args):
        wx.GraphicsContext.SetPen(*args)
    
    @staticmethod
    def setFont(context, font, brush):
	return context.SetFont(font, brush.Colour)
    
    
    @staticmethod
    def strokePath(*args):
        return wx.GraphicsContext.StrokePath(*args)
    
    @staticmethod
    def translate(*args):
        return wx.GraphicsContext.Translate(*args)
    
    @staticmethod
    def DrawText(context, text, x, y, brush=NullGraphicsBrush, anchor='start'):
        #SVG spec appears to originate text from the bottom
        #rather than the top as with our API. This function
        #will measure and then re-orient the text as needed.
        w, h = context.GetTextExtent(text)
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
