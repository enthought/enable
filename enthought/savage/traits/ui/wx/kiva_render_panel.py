import time

from enthought.savage.svg.backends.kiva import renderer

class RenderPanel(renderer.Canvas):
    def __init__(self, parent, document=None):
        renderer.Canvas.__init__(self, parent)
        self.lastRender = None
        self.document = document
        self.zoom = 100
#        self.offset = wx.Point(0,0)

    def do_draw(self, gc):
        start = time.time()
        gc.clear()

        if not self.document:
            gc.show_text_at_point("No Document", 20, 20)
            return

        # SVG origin is upper right with y positive is down. argh.
        # Set up the transforms to fix this up.
        gc.translate_ctm(0, gc.height())
        # zoom percentage
        scale = float(self.zoom) / 100.0
        gc.scale_ctm(scale, -scale)
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

    def Refresh(self):
        # There is a strange bug here: if neither the dirty flag is set nor the
        # draw method called, the next update will not be drawn. If only one of
        # them is done, then the next draw will be upside down. This feels like
        # a wx bug. Its a bit wasteful, since Refresh is supposed to queue the
        # paint, but I don't see another way around it right now.
        self.dirty = True
        self.draw()
        super(RenderPanel, self).Refresh()
