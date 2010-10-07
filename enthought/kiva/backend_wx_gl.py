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

""" The WX OpenGL backend.

    This backend only uses OpenGL to get a device to draw onto. AGG is used for
    the actual drawing.
"""


__all__ = ["GraphicsContext", "Canvas", "CanvasWindow"]

import time
import sys
import wx


if float(wx.__version__[:3]) >= 2.5:
    from wxPython import wx as oldwx
    from wx import glcanvas
else:
    oldwx = wx
    # ??? - Let's hope and pray this works.  Utterly untested.
    glcanvas = wx.glcanvas


from backend_image import GraphicsContextSystem as GraphicsContext
from agg import plat_support, CompiledPath
from backend_wx import BaseWxCanvas

def font_metrics_provider():
    """ Create an object to be used for querying font metrics.
    """

    return GraphicsContext((1, 1))


class Canvas(BaseWxCanvas, glcanvas.GLCanvas):
    "A Canvas that uses the Wx GLCanvas"
    def __init__(self, parent, id = -1, size = wx.DefaultSize):
        glcanvas.GLCanvas.__init__(self, parent, id, wx.Point(0, 0), size, 
                             wx.SUNKEN_BORDER | wx.WANTS_CHARS | \
                             wx.FULL_REPAINT_ON_RESIZE )
        super(Canvas, self).__init__()

    def _create_kiva_gc(self, size):
        return GraphicsContext(size)

    def blit(self, event):
        t1 = time.time()
        wdc = wx.PaintDC(self)
        self.gc.pixel_map.draw_to_glcanvas(0, 0)
        self.SwapBuffers()
        t2 = time.time()
        self.blit_time = t2-t1
        self.dirty = False
        return

    def OnSize(self,event):
        # resize buffer bitmap and repaint.
        sz = self.GetClientSizeTuple()
        if (sz != (self.gc.width(),self.gc.height()) and self.GetContext()):
            self.SetCurrent()
            plat_support.resize_gl(*sz)
            self.new_gc(sz)
        event.Skip()
        return

    def OnPaint(self,event):
        self.SetCurrent()
        self.paint(event)
        return

    def Refresh(self):
        self.paint(None)
        glcanvas.GLCanvas.Refresh(self)
        return



class _CanvasWindow(wx.Frame):
    def __init__(self, id=-1, title='Kiva Canvas',size=(600,800),
                 canvas_class=Canvas):
        parent = oldwx.NULL
        wx.Frame.__init__(self, parent,id,title, size=size)
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
