import time
import wx

from svg.backends.wx import renderer

class RenderPanel(wx.PyPanel):
    def __init__(self, parent, document=None):
        wx.PyPanel.__init__(self, parent)
        self.lastRender = None
        self.document = document
        self.zoom = 100
        self.offset = wx.Point(0,0)
        self.SetBackgroundStyle(wx.BG_STYLE_CUSTOM)
        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.Bind(wx.EVT_MOUSEWHEEL, self.OnWheel)
        self.Bind(wx.EVT_MOTION, self.OnMotion)
        self.Bind(wx.EVT_LEFT_DOWN, self.OnLeftDown)
        self.Bind(wx.EVT_LEFT_UP, self.OnLeftUp)
        self.Bind(wx.EVT_MIDDLE_UP, self.OnMiddleClick)
 
    def OnPaint(self, evt):
        start = time.time()
        dc = wx.BufferedPaintDC(self)
        dc.SetBackground(wx.Brush(self.GetBackgroundColour()))
        dc.Clear()
        if not self.document:
            dc.DrawText("No Document", 20, 20)
            return
        gc = wx.GraphicsContext_Create(dc)
        scale = float(self.zoom) / 100.0

        gc.Translate(*self.offset)
        gc.Scale(scale, scale)
 
        self.document.render(gc)
        self.lastRender = time.time() - start

    def GetBestSize(self):
        if not self.document:
            return (-1,-1)
        sz = map(int,self.document.tree.getroot().get("viewBox").split())
        return wx.Rect(*sz).GetSize()

    def OnWheel(self, evt):
        self.zoom += (evt.m_wheelRotation / evt.m_wheelDelta) * 10
        self.Refresh()

    def OnLeftDown(self, evt):
        self.SetCursor(wx.StockCursor(wx.CURSOR_HAND))
        self.CaptureMouse()
        self.offsetFrom = evt.GetPosition()
        evt.Skip()

    def OnLeftUp(self, evt):
        if self.HasCapture():
            self.ReleaseMouse()
        self.SetCursor(wx.NullCursor)
        evt.Skip()

    def OnMotion(self, evt):
        if not self.HasCapture():
            return
        self.offset += (evt.GetPosition() - self.offsetFrom)
        self.offsetFrom = evt.GetPosition()
        self.Refresh()

    def OnMiddleClick(self, evt):
        self.offset = wx.Point(0,0)
        self.zoom = 100
        self.Refresh()
