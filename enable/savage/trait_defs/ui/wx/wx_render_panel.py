# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
import time

import wx

from enable.savage.svg.backends.wx import renderer
from traitsui.wx.constants import WindowColor


class RenderPanel(wx.Panel):
    def __init__(self, parent, document=None):
        super(RenderPanel, self).__init__(parent)
        self.lastRender = None
        self.document = document
        self.zoom_x = 100
        self.zoom_y = 100
        self.offset = wx.Point(0, 0)
        self.SetBackgroundStyle(wx.BG_STYLE_CUSTOM)
        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.Bind(wx.EVT_MOUSEWHEEL, self.OnWheel)
        self.Bind(wx.EVT_MOTION, self.OnMotion)
        self.Bind(wx.EVT_LEFT_DOWN, self.OnLeftDown)
        self.Bind(wx.EVT_LEFT_UP, self.OnLeftUp)
        self.Bind(wx.EVT_MIDDLE_UP, self.OnMiddleClick)
        self.Bind(wx.EVT_ENTER_WINDOW, self.OnEnterWindow)
        self.Bind(wx.EVT_LEAVE_WINDOW, self.OnLeaveWindow)

    def OnPaint(self, evt):
        start = time.time()
        dc = wx.BufferedPaintDC(self)
        dc.SetBackground(wx.Brush(self.GetBackgroundColour()))
        dc.Clear()
        if not self.document:
            dc.DrawText("No Document", 20, 20)
            return
        gc = wx.GraphicsContext_Create(dc)

        gc.Translate(*self.offset)
        gc.Scale(float(self.zoom_x) / 100, float(self.zoom_y) / 100)

        self.document.render(gc)
        self.lastRender = time.time() - start

    def GetBackgroundColour(self):
        return WindowColor

    def GetBestSize(self):
        if not self.document:
            return (-1, -1)

        return wx.Size(*(self.document.getSize()))

    def OnWheel(self, evt):
        delta = (evt.m_wheelRotation / evt.m_wheelDelta) * 10
        self.zoom_x += delta
        self.zoom_y += delta
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
        self.offset += evt.GetPosition() - self.offsetFrom
        self.offsetFrom = evt.GetPosition()
        self.Refresh()

    def OnMiddleClick(self, evt):
        self.offset = wx.Point(0, 0)
        self.zoom_x = 100
        self.zoom_y = 100
        self.Refresh()

    def OnEnterWindow(self, evt):
        pass

    def OnLeaveWindow(self, evt):
        pass
