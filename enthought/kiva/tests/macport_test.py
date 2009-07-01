#!/usr/bin/env pythonw

import sys

if sys.platform == 'darwin':
    import wx

    from enthought.kiva.mac import get_macport

    class SimpleWindow(wx.Frame):
        """
        Simple test of get_macport().
        """
        def __init__(self):
            wx.Frame.__init__(self, parent=None, id=-1, title="foo", 
                               pos=(100,100),
                               size=(300,300))
            oldstyle = self.GetWindowStyle()
            oldstyle = oldstyle | wx.FULL_REPAINT_ON_RESIZE
            self.SetWindowStyle(oldstyle)
            self.Show(1)
            self.Bind(wx.EVT_PAINT, self.OnPaint)
            self.memdc = wx.MemoryDC()
            self.bitmap = wx.EmptyBitmap(200,200)
            self.memdc.SelectObject(self.bitmap)
            print "constructed frame"

        def OnPaint(self, evt):
            dc = wx.PaintDC(self)
            print "paintdc.this:", dc.this
            print "paintdc.macport:", get_macport(dc)
            print "memdc.this:", self.memdc.this
            print "memdc.macport:", get_macport(self.memdc)

    class MyApp(wx.App):
        def OnInit(self):
            w = SimpleWindow()
            return 1

    app = MyApp(False)
    app.MainLoop()
