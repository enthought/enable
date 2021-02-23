# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
import svg

import wx
import wx.py.shell
import wx.aui
import wx.grid


class PathGrid(wx.grid.Grid):
    def __init__(self, parent, pathOps):
        super(PathGrid, self).__init__(parent)
        self.CreateGrid(100, 10)
        firstColAttr = wx.grid.GridCellAttr()
        choices = sorted(pathOps.keys())
        firstColAttr.SetEditor(wx.grid.GridCellChoiceEditor(choices))
        self.SetColMinimalWidth(0, 140)
        self.SetColAttr(0, firstColAttr)


class PathPanel(wx.Panel):
    ctx = None
    path = None

    def __init__(self, parent, contextSource):
        super(PathPanel, self).__init__(
            parent, style=wx.FULL_REPAINT_ON_RESIZE
        )
        self.contextSource = contextSource
        self.Bind(wx.EVT_PAINT, self.OnPaint)

    def GetContext(self):
        self.ctx = self.contextSource(self)

    def SetPath(self, path):
        self.ctx = self.contextSource(self)
        self.path = path
        self.Update()
        self.Refresh()

    def OnPaint(self, evt):
        dc = wx.PaintDC(self)
        self.GetContext()
        if not (self.ctx and self.path):
            return
        self.ctx.DrawPath(self.path)


class DrawFrame(wx.Frame):
    def __init__(self, parent, *args, **kwargs):
        wx.Frame.__init__(self, parent, *args, **kwargs)
        self.pathOps = {
            k: v
            for (k, v) in wx.GraphicsPath.__dict__.items()
            if k.startswith("Add")
        }
        self.pathOps["CloseSubpath"] = wx.GraphicsPath.CloseSubpath
        self.pathOps["MoveToPoint"] = wx.GraphicsPath.MoveToPoint
        self.pathOps[""] = None

        self._mgr = wx.aui.AuiManager()
        self._mgr.SetManagedWindow(self)

        self.panel = PathPanel(self, self.CreateContext)

        self.locals = {
            "wx": wx,
            "frame": self,
            "panel": self.panel,
            "context": None,
        }

        self.nb = wx.aui.AuiNotebook(self)

        self.shell = wx.py.shell.Shell(self.nb, locals=self.locals)

        self.grid = PathGrid(self.nb, self.pathOps)

        self.nb.AddPage(self.shell, "Shell")
        self.nb.AddPage(self.grid, "Path", True)

        self._mgr.AddPane(
            self.nb,
            wx.aui.AuiPaneInfo()
            .Bottom()
            .CaptionVisible(False)
            .BestSize((-1, 300)),
        )
        self._mgr.AddPane(self.panel, wx.aui.AuiPaneInfo().CenterPane())

        self._mgr.Update()

        wx.CallAfter(self.panel.GetContext)
        # wx.CallAfter(self.shell.SetFocus)
        self.Bind(wx.grid.EVT_GRID_CELL_CHANGE, self.OnPathChange)

    def CreateContext(self, target):
        ctx = wx.GraphicsContext_Create(target)
        ctx.SetPen(wx.BLACK_PEN)
        ctx.SetBrush(wx.RED_BRUSH)
        self.locals["context"] = ctx
        return ctx

    def OnPathChange(self, evt):
        path = wx.GraphicsRenderer_GetDefaultRenderer().CreatePath()
        self.FillPath(path)
        self.panel.SetPath(path)

    def FillPath(self, path):
        for row in range(100):
            operation = self.grid.GetCellValue(row, 0)
            if not operation:
                return
            args = []
            for col in range(1, 20):
                v = self.grid.GetCellValue(row, col)
                if not v:
                    break
                args.append(float(v))
            self.pathOps[operation](path, *args)


if __name__ == "__main__":
    app = wx.App(False)
    frame = DrawFrame(None, size=(800, 600))
    frame.Centre()
    frame.Show()
    app.MainLoop()
