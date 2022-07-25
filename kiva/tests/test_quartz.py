# (C) Copyright 2005-2022 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
import sys
import unittest

from kiva.tests._testing import skip_if_not_wx


class TestQuartz(unittest.TestCase):
    def test_quartz_importable(self):
        if sys.platform != "darwin":
            from unittest.case import SkipTest

            raise SkipTest("quartz is only built on OS X")

        from kiva.quartz import ABCGI
        from kiva.quartz import CTFont
        from kiva.quartz import mac_context
        del ABCGI
        del CTFont
        del mac_context

    @skip_if_not_wx
    def test_macport(self):
        if sys.platform != "darwin":
            from unittest.case import SkipTest

            raise SkipTest("macport is only built on OS X")

        import wx

        from kiva.quartz import get_macport

        class SimpleWindow(wx.Frame):
            """
            Simple test of get_macport().
            """

            def __init__(self):
                wx.Frame.__init__(
                    self,
                    parent=None,
                    id=-1,
                    title="foo",
                    pos=(100, 100),
                    size=(300, 300),
                )
                oldstyle = self.GetWindowStyle()
                oldstyle = oldstyle | wx.FULL_REPAINT_ON_RESIZE
                self.SetWindowStyle(oldstyle)
                self.Show(1)
                self.Bind(wx.EVT_PAINT, self.OnPaint)
                self.memdc = wx.MemoryDC()
                self.bitmap = wx.EmptyBitmap(200, 200)
                self.memdc.SelectObject(self.bitmap)

            def OnPaint(self, evt):
                dc = wx.PaintDC(self)
                print("paintdc.this:", dc.this)
                print("paintdc.macport: %x" % get_macport(dc))
                print("memdc.this:", self.memdc.this)
                print("memdc.macport: %x" % get_macport(self.memdc))

                # We're done here
                self.Close()

        class MyApp(wx.App):
            def OnInit(self):
                SimpleWindow()
                return 1

        app = MyApp(False)
        app.MainLoop()

    def test_font_names(self):
        from kiva.quartz.CTFont import default_font_info

        names = default_font_info.names()

        # regression test for Enable#964
        self.assertFalse(any(s.startswith("b'") for s in names))
