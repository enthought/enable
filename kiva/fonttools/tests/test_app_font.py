# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
import os
import unittest

import pkg_resources

from traits.etsconfig.api import ETSConfig

from kiva.api import add_application_fonts, Font

is_null = (ETSConfig.toolkit in ("", "null"))
is_qt = ETSConfig.toolkit.startswith("qt")
is_wx = (ETSConfig.toolkit == "wx")
data_dir = pkg_resources.resource_filename("kiva.fonttools.tests", "data")


@unittest.skipIf(not is_null, "Test only for null toolkit")
class TestNullApplicationFonts(unittest.TestCase):
    def test_add_application_font(self):
        path = os.path.join(data_dir, "TestTTF.ttf")
        family = "Test TTF"
        kivafont = Font(family)

        # Before adding the font
        with self.assertWarns(UserWarning):
            self.assertNotEqual(kivafont.findfont().filename, path)

        add_application_fonts([path])

        # After adding the font
        self.assertEqual(kivafont.findfont().filename, path)


@unittest.skipIf(not is_qt, "Test only for qt")
class TestQtApplicationFonts(unittest.TestCase):
    def setUp(self):
        from pyface.qt import QtGui

        application = QtGui.QApplication.instance()
        if application is None:
            self.application = QtGui.QApplication([])
        else:
            self.application = application
        unittest.TestCase.setUp(self)

    def test_add_application_font(self):
        from pyface.qt import QtGui

        path = os.path.join(data_dir, "TestTTF.ttf")
        family = "Test TTF"
        font_db = QtGui.QFontDatabase()

        # Before adding the font
        self.assertNotIn(family, font_db.families())

        add_application_fonts([path])

        # After adding the font
        self.assertIn(family, font_db.families())


@unittest.skipIf(not is_wx, "Test only for wx")
class TestWxApplicationFonts(unittest.TestCase):
    def setUp(self):
        import wx

        application = wx.App.Get()
        if application is None:
            self.application = wx.App()
        else:
            self.application = application
        unittest.TestCase.setUp(self)

    # XXX: How do we check to see if Wx loaded our font?
    @unittest.expectedFailure
    def test_add_application_font(self):
        import wx

        path = os.path.join(data_dir, "TestTTF.ttf")
        family = "Test TTF"

        fontinfo = wx.FontInfo()
        fontinfo.FaceName(family)
        wxfont = wx.Font(fontinfo)

        # Before adding the font
        self.assertFalse(wxfont.IsOk())

        add_application_fonts([path])

        # After adding the font
        self.assertTrue(wxfont.IsOk())
