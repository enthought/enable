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
import sys
import unittest

from pkg_resources import resource_filename

from .._scan_sys import scan_system_fonts, scan_user_fonts

data_dir = resource_filename("kiva.fonttools.tests", "data")
is_macos = (sys.platform == "darwin")
is_windows = (sys.platform in ("win32", "cygwin"))
is_generic = not (is_macos or is_windows)


class TestFontDirectoryScanning(unittest.TestCase):
    def test_directory_scanning(self):
        expected = [
            os.path.join(data_dir, fname)
            for fname in os.listdir(data_dir)
            if os.path.splitext(fname)[-1] in (".ttf", ".ttc")
        ]
        fonts = scan_system_fonts(data_dir, fontext="ttf")
        self.assertListEqual(sorted(expected), sorted(fonts))

        expected = [
            os.path.join(data_dir, fname)
            for fname in os.listdir(data_dir)
            if os.path.splitext(fname)[-1] == ".afm"
        ]
        fonts = scan_system_fonts(data_dir, fontext="afm")
        self.assertListEqual(sorted(expected), sorted(fonts))

    def test_directories_scanning(self):
        expected = sorted([
            os.path.join(data_dir, fname)
            for fname in os.listdir(data_dir)
            if os.path.splitext(fname)[-1] in (".ttf", ".ttc")
        ])
        # Pass a list of directories instead of a single path string
        fonts = scan_system_fonts([data_dir], fontext="ttf")
        self.assertListEqual(sorted(expected), sorted(fonts))

    def test_user_font_scanning(self):
        ttf_fonts = scan_user_fonts(data_dir, fontext="ttf")
        self.assertEqual(len(ttf_fonts), 3)

        afm_fonts = scan_user_fonts(data_dir, fontext="afm")
        self.assertEqual(len(afm_fonts), 1)

    @unittest.skipIf(not is_generic, "This test is only for generic platforms")
    def test_generic_scanning(self):
        fonts = scan_system_fonts(fontext="ttf")
        self.assertNotEqual([], fonts)

    @unittest.skipIf(not is_macos, "This test is only for macOS")
    def test_macos_scanning(self):
        fonts = scan_system_fonts(fontext="ttf")
        self.assertNotEqual([], fonts)

    @unittest.skipIf(not is_windows, "This test is only for Windows")
    def test_windows_scanning(self):
        fonts = scan_system_fonts(fontext="ttf")
        self.assertNotEqual([], fonts)
