#------------------------------------------------------------------------------
# Copyright (c) 2005, Enthought, Inc.
# some parts copyright 2002 by Space Telescope Science Institute
# All rights reserved.
# 
# This software is provided without warranty under the terms of the BSD
# license included in enthought/LICENSE.txt and may be redistributed only
# under the conditions described in the aforementioned license.  The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
# Thanks for using Enthought open source!
#------------------------------------------------------------------------------

"""
This is the WX backend that draws into a normal WX graphics context using
the Image backend.  It detects the OS type and creates the appropriate
low-level support classes to enable Agg to draw into a WX memory buffer.
"""


# This automatically determines which platform-dependent WX backend should be
# loaded and exports the appropriate Canvas and CanvasWindow.

__all__ = ["GraphicsContext", "Canvas", "CanvasWindow", "CompiledPath",
    "font_metrics_provider"]

# Standard library imports.
import sys
from time import clock as now

# Major library imports.
import wx

# Local imports
from fonttools import Font

WidgetClass = wx.Window


class BaseWxCanvas(object):
    """
    All the WX backends ultimately use the Agg backend to do the actual drawing.
    However, they all differ in how they construct a memory buffer to hand to
    Agg.  This class defines the basic functionality expected of a Kiva Wx backend
    canvas.  Its concrete methods handle some of the interaction with wx, but
    it needs to be mixed in with an actual Wx drawable (e.g Window,
    Frame, GLCanvas, etc.).
    
    Subclasses must implement _create_kiva_gc and blit, and they will most
    likely provide their own __init__ that calls the Wx drawable-specific
    base constructor as well as this class's.
    """
    def __init__(self):
        wx.EVT_PAINT(self, self.OnPaint)
        wx.EVT_SIZE(self, self.OnSize)      
        wx.EVT_ERASE_BACKGROUND(self, self.OnErase)
        self.gc = None
        self.new_gc()

        self.blit_time = 0.01 # a decent guess for first call
        self.total_paint_time = 0.0
        self.clear_color = (1,1,1)
        self._do_draw = 0
        self._win_id = None
        self._dc = None
        return

    def _create_kiva_gc(self, size):
        """
        Returns a new backend-dependent GraphicsContext* instance of the
        given size.
        """
        raise NotImplementedError

    def blit(self, event):
        """
        The actual drawing call.  event is the wxEvent that triggered the
        paint/referesh call that resulted in blit being called.
        """
        raise NotImplementedError

    def size(self):
        return self.GetSizeTuple()    
    
    def paint(self, event):
        tt1 = now()
        if self.dirty:
            self.clear()
            self.draw()
        self.blit(event)
        tt2 = now()
        self.total_paint_time = tt2 - tt1
        return

    def new_gc(self, sz = None):
        if sz is None:
            sz = self.GetClientSizeTuple()
            if (sz[0] <= 0) or (sz[1] <= 0):
                sz = (100, 100)
        t1 = now()
        if self.gc is not None:
            del self.gc
            self.gc = None   # make sure self.gc is defined
        self.gc = self._create_kiva_gc(sz)
        t2 = now()
        self.new_gc_time = t2-t1
        self.dirty = 1
        return

    def clear(self):
        t1 = now()
        self.gc.clear(self.clear_color)
        t2 = now()
        self.clear_time = t2-t1
        return

    def draw(self):
        t1 = now()
        self.do_draw(self.gc)
        t2 = now()
        self.draw_time = t2-t1
        return
        
    def OnErase(self,event):
        pass
    
    def OnSize(self,event):
        # resize buffer bitmap and repaint.
        sz = self.GetClientSizeTuple()
        if sz != (self.gc.width(),self.gc.height()):
            self.new_gc(sz)
        event.Skip()
        return

    def OnPaint(self,event):
        self.paint(event)
        return

    def OnIdle(self, event):
        self.paint(event)
        self._do_draw = 0
        return
        


# Define a different base class depending on the platform.

if sys.platform == 'darwin':
    from mac import get_macport, ABCGI
    from mac.ABCGI import CGBitmapContext, CGImage, CGImageFile, \
        CGLayerContext, CGMutablePath

    # The Mac backend only supports numpy.
    import numpy as np

    if wx.VERSION[:2] == (2, 6):
        def gc_for_dc(dc):
            """ Return the CGContext corresponding to the given wx.DC.
            """
            port = get_macport(dc)
            return ABCGI.CGContextForPort(port)

    elif wx.VERSION[:2] == (2, 8):
        class UnflippingCGContext(ABCGI.CGContextInABox):
            """ Vertically flip the context to undo wx's flipping.
            """

            def __init__(self, *args, **kwds):
                ABCGI.CGContextInABox.__init__(self, *args, **kwds)
                self._begun = False

            def begin(self):
                if self._begun:
                    return
                self.save_state()
                self.translate_ctm(0, self.height())
                self.scale_ctm(1.0, -1.0)
                self._begun = True

            def end(self):
                if self._begun:
                    self.restore_state()
                    self._begun = False

        def gc_for_dc(dc):
            """ Return the CGContext corresponding to the given wx.DC.
            """
            pointer = get_macport(dc)
            gc = UnflippingCGContext(pointer, dc.GetSizeTuple())
            return gc


    CompiledPath = CGMutablePath
    Image = CGImageFile

    class GraphicsContext(CGLayerContext):
        def __init__(self, size_or_array, *args, **kwds):
            # Create a tiny base context to spawn the CGLayerContext from.
            bitmap = CGBitmapContext((1,1))
            if isinstance(size_or_array, np.ndarray):
                # Initialize the layer with an image.
                image = CGImage(size_or_array)
                width = image.width
                height = image.height
            else:
                # No initialization.
                image = None
                width, height = size_or_array
            CGLayerContext.__init__(self, bitmap, 
                (width, height))
            if image is not None:
                self.draw_image(image)

    class Canvas(BaseWxCanvas, WidgetClass):
        """ Mac wx Kiva canvas.
        """
        def __init__(self, parent, id = 01, size = wx.DefaultSize):
            # need to init self.memDC before calling BaseWxCanvas.__init__ 
            self.memDC = wx.MemoryDC()
            self.size = (size.GetWidth(), size.GetHeight())
            WidgetClass.__init__(self, parent, id, wx.Point(0, 0), size, 
                                 wx.SUNKEN_BORDER | wx.WANTS_CHARS | \
                                 wx.FULL_REPAINT_ON_RESIZE )
            BaseWxCanvas.__init__(self)
            return
        
        def _create_kiva_gc(self, size):
            self.size = size
            self.bitmap = wx.EmptyBitmap(size[0], size[1])
            self.memDC.SelectObject(self.bitmap)
            gc = gc_for_dc(self.memDC)
            #gc.begin()
            #print " **** gc is:", gc
            return gc
        
        def blit(self, event):
            t1 = now()
            paintdc = wx.PaintDC(self)
            paintdc.Blit(0, 0, self.size[0], self.size[1],
                         self.memDC, 0, 0)
            t2 = now()
            self.blit_time = t2 - t1
            self.dirty = 0
            return
            
        def draw(self):
            t1 = now()
            self.gc.begin()
            self.do_draw(self.gc)
            self.gc.end()
            t2 = now()
            self.draw_time = t2-t1
            return
        

else:
    # the GraphicsContextSystem stuff should eventually be moved out of the
    # image backend.
    from backend_image import GraphicsContextSystem as GraphicsContext
    from agg import CompiledPath

    class Canvas(BaseWxCanvas, WidgetClass):
        "The basic wx Kiva canvas."
        def __init__(self, parent, id = -1, size = wx.DefaultSize):
            WidgetClass.__init__(self, parent, id, wx.Point(0, 0), size, 
                                 wx.SUNKEN_BORDER | wx.WANTS_CHARS | \
                                 wx.FULL_REPAINT_ON_RESIZE )
            BaseWxCanvas.__init__(self)
            return
    
        def _create_kiva_gc(self, size):
            return GraphicsContext(size)
    
        def blit(self, event):
            t1 = now()
    
            if self._dc is None:
                self._dc = wx.PaintDC(self)
            self.gc.pixel_map.draw_to_wxwindow(self, 0, 0)
            self._dc = None
    
            t2 = now()
            self.blit_time = t2-t1
            self.dirty = 0
            return

def font_metrics_provider():
    gc = GraphicsContext((1, 1))
    gc.set_font(Font())
    return gc

class _CanvasWindow(wx.Frame):
    def __init__(self, id=-1, title='Kiva Canvas', size=(600,800),
                 canvas_class=Canvas):
        parent = None
        wx.Frame.__init__(self, parent, id, title, size=size)
        self.canvas = canvas_class(self)
        self.Show(1)
        return

CanvasWindow = _CanvasWindow
try:
    from numpy import which
    if which[0]!='numpy':
        import gui_thread
        CanvasWindow = gui_thread.register(_CanvasWindow)
except ImportError:
    pass
