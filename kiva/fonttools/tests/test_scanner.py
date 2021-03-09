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
from unittest import mock

from fontTools.ttLib import TTFont
from pkg_resources import resource_filename

from .._scanner import (
    _afm_font_property, _build_afm_entries, _ttf_font_property,
    create_font_list, FontEntry, scan_system_fonts
)

data_dir = resource_filename("kiva.fonttools.tests", "data")
is_macos = (sys.platform == "darwin")
is_windows = (sys.platform in ("win32", "cygwin"))
is_generic = not (is_macos or is_windows)


class TestFontEntryCreation(unittest.TestCase):
    def setUp(self):
        self.ttc_fontpath = os.path.join(data_dir, "TestTTC.ttc")
        self.ttf_fontpath = os.path.join(data_dir, "TestTTF.ttf")

    def test_fontlist_duplicates(self):
        # When
        three_duplicate_ttfs = [self.ttf_fontpath] * 3
        fontlist = create_font_list(three_duplicate_ttfs)

        # Then
        self.assertEqual(len(fontlist), 1)
        self.assertIsInstance(fontlist[0], FontEntry)

    def test_fontlist_from_ttc(self):
        # When
        fontlist = create_font_list([self.ttc_fontpath])

        # Then
        self.assertEqual(len(fontlist), 2)
        for fontprop in fontlist:
            self.assertIsInstance(fontprop, FontEntry)

    @mock.patch("kiva.fonttools._scanner._ttf_font_property",
                side_effect=ValueError)
    def test_ttc_exception_on__ttf_font_property(self, m_ttf_font_property):
        # When
        with self.assertLogs("kiva"):
            fontlist = create_font_list([self.ttc_fontpath])

        # Then
        self.assertEqual(len(fontlist), 0)
        self.assertEqual(m_ttf_font_property.call_count, 1)

    @mock.patch("kiva.fonttools._scanner.TTCollection",
                side_effect=RuntimeError)
    def test_ttc_exception_on_TTCollection(self, m_TTCollection):
        # When
        with self.assertLogs("kiva"):
            fontlist = create_font_list([self.ttc_fontpath])

        # Then
        self.assertEqual(len(fontlist), 0)
        self.assertEqual(m_TTCollection.call_count, 1)


class TestFontDirectoryScanning(unittest.TestCase):
    def test_directory_scanning(self):
        expected = [
            os.path.join(data_dir, fname)
            for fname in os.listdir(data_dir)
        ]
        fonts = scan_system_fonts(data_dir, fontext="ttf")
        self.assertListEqual(sorted(expected), sorted(fonts))

        # There are no AFM fonts in the test data
        fonts = scan_system_fonts(data_dir, fontext="afm")
        self.assertListEqual([], fonts)

    def test_directories_scanning(self):
        expected = sorted([
            os.path.join(data_dir, fname)
            for fname in os.listdir(data_dir)
        ])
        # Pass a list of directories instead of a single path string
        fonts = scan_system_fonts([data_dir], fontext="ttf")
        self.assertListEqual(sorted(expected), sorted(fonts))

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


class TestAFMFontEntry(unittest.TestCase):
    def test_afm_font_failure(self):
        # We have no AFM fonts, so test some error handling
        with self.assertLogs("kiva"):
            entries = _build_afm_entries("nonexistant.path")
        self.assertListEqual([], entries)

        # XXX: Once AFM code has been converted to expect bytestrings:
        # Add a test which passes an existing file (non-afm) to
        # _build_afm_entries.

    def test_property_branches(self):
        fake_path = os.path.join(data_dir, "TestAFM.afm")

        class FakeAFM:
            def __init__(self, name, family, angle, weight):
                self.name = name
                self.family = family
                self.angle = angle
                self.weight = weight

            def get_angle(self):
                return self.angle

            def get_familyname(self):
                return self.family

            def get_fontname(self):
                return self.name

            def get_weight(self):
                return self.weight

        # Given
        fake_font = FakeAFM("TestyFont", "Testy", 0, "Bold")
        exp_name = "Testy"
        exp_style = "normal"
        exp_variant = "normal"
        exp_weight = 700
        exp_stretch = "normal"
        exp_size = "scalable"
        # When
        entry = _afm_font_property(fake_path, fake_font)
        # Then
        self.assertEqual(entry.name, exp_name)
        self.assertEqual(entry.style, exp_style)
        self.assertEqual(entry.variant, exp_variant)
        self.assertEqual(entry.weight, exp_weight)
        self.assertEqual(entry.stretch, exp_stretch)
        self.assertEqual(entry.size, exp_size)

        # Style variations
        italics = (
            FakeAFM("TestyFont", "Testy Italic", 0, "Bold"),
            FakeAFM("TestyFont", "Testy Fancy", 30, "Bold"),
        )
        oblique = FakeAFM("TestyFont", "Testy Oblique", 0, "Bold")
        # Oblique
        entry = _afm_font_property(fake_path, oblique)
        self.assertEqual(entry.style, "oblique")
        # Italic
        for font in italics:
            entry = _afm_font_property(fake_path, font)
            self.assertEqual(entry.style, "italic")

        # Given
        fake_font = FakeAFM("TestyFont", "Testy Capitals", 0, "Bold")
        exp_variant = "small-caps"
        # When
        entry = _afm_font_property(fake_path, fake_font)
        # Then
        self.assertEqual(entry.variant, exp_variant)

        # Given
        stretches = {
            "condensed": FakeAFM("TestyFont Narrow", "Testy", 0, "Regular"),
            "expanded": FakeAFM("Testy Wide", "Testy", 0, "Light"),
            "semi-condensed": FakeAFM("Testy Demi Cond", "Testy", 0, "Light"),
        }
        for stretch, font in stretches.items():
            # When
            entry = _afm_font_property(fake_path, font)
            # Then
            self.assertEqual(entry.stretch, stretch)


class TestTTFFontEntry(unittest.TestCase):
    def test_font(self):
        # Given
        test_font = os.path.join(data_dir, "TestTTF.ttf")
        exp_name = "Test TTF"
        exp_style = "normal"
        exp_variant = "normal"
        exp_weight = 400
        exp_stretch = "normal"
        exp_size = "scalable"

        # When
        entry = _ttf_font_property(test_font, TTFont(test_font))

        # Then
        self.assertEqual(entry.name, exp_name)
        self.assertEqual(entry.style, exp_style)
        self.assertEqual(entry.variant, exp_variant)
        self.assertEqual(entry.weight, exp_weight)
        self.assertEqual(entry.stretch, exp_stretch)
        self.assertEqual(entry.size, exp_size)

    def test_font_with_italic_style(self):
        """Test that a font with Italic style, writing with a capital
        "I" is correctly identified as "italic" style.
        """
        # Given
        test_font = os.path.join(data_dir, "TestTTF Italic.ttf")
        exp_name = "Test TTF"
        exp_style = "italic"
        exp_variant = "normal"
        exp_weight = 400
        exp_stretch = "normal"
        exp_size = "scalable"

        # When
        entry = _ttf_font_property(test_font, TTFont(test_font))

        # Then
        self.assertEqual(entry.name, exp_name)
        self.assertEqual(entry.style, exp_style)
        self.assertEqual(entry.variant, exp_variant)
        self.assertEqual(entry.weight, exp_weight)
        self.assertEqual(entry.stretch, exp_stretch)
        self.assertEqual(entry.size, exp_size)

    def test_nameless_font(self):
        # Given
        test_font = os.path.join(data_dir, "TestTTF.ttf")

        # When
        target = "kiva.fonttools._scanner.get_ttf_prop_dict"
        with mock.patch(target, return_value={}):
            with self.assertRaises(KeyError):
                # Pass None since we're mocking get_ttf_prop_dict
                _ttf_font_property(test_font, None)

    def test_property_branches(self):
        target = "kiva.fonttools._scanner.get_ttf_prop_dict"
        test_font = os.path.join(data_dir, "TestTTF.ttf")

        # Given
        exp_name = "TestyFont Bold Capitals"
        prop_dict = {
            "name": exp_name,
            "sfnt4": exp_name,
        }
        # When
        with mock.patch(target, return_value=prop_dict):
            # Pass None since we're mocking get_ttf_prop_dict
            entry = _ttf_font_property(test_font, None)
        # Then
        self.assertEqual(entry.variant, "small-caps")

        # Given
        exp_name = "TestyFont Bold Oblique"
        prop_dict = {
            "name": exp_name,
            "sfnt4": exp_name,
        }
        # When
        with mock.patch(target, return_value=prop_dict):
            # Pass None since we're mocking get_ttf_prop_dict
            entry = _ttf_font_property(test_font, None)
        # Then
        self.assertEqual(entry.style, "oblique")

        stretch_options = {
            "TestyFont Narrow": "condensed",
            "TestyFont Condensed": "condensed",
            "TestyFont Demi Cond": "semi-condensed",
            "TestyFont Wide": "expanded",
            "TestyFont Expanded": "expanded",
        }
        for name, stretch in stretch_options.items():
            # Given
            prop_dict = {
                "name": name,
                "sfnt4": name,
            }
            # When
            with mock.patch(target, return_value=prop_dict):
                # Pass None since we're mocking get_ttf_prop_dict
                entry = _ttf_font_property(test_font, None)
            # Then
            self.assertEqual(entry.stretch, stretch)
