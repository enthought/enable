# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
import contextlib
import importlib
import os
import sys
import tempfile
import unittest
from unittest import mock

from pkg_resources import resource_filename
from fontTools.ttLib import TTFont

from traits.etsconfig.api import ETSConfig

from .. import font_manager as font_manager_module
from ..font_manager import (
    createFontList, default_font_manager, FontEntry, FontManager,
    ttfFontProperty,
)
from ._testing import patch_global_font_manager

data_dir = resource_filename("kiva.fonttools.tests", "data")


class TestCreateFontList(unittest.TestCase):
    def setUp(self):
        self.ttc_fontpath = os.path.join(data_dir, "TestTTC.ttc")

    def test_fontlist_from_ttc(self):
        # When
        fontlist = createFontList([self.ttc_fontpath])

        # Then
        self.assertEqual(len(fontlist), 2)
        for fontprop in fontlist:
            self.assertIsInstance(fontprop, FontEntry)

    @mock.patch("kiva.fonttools.font_manager.ttfFontProperty",
                side_effect=ValueError)
    def test_ttc_exception_on_ttfFontProperty(self, m_ttfFontProperty):
        # When
        with self.assertLogs("kiva"):
            fontlist = createFontList([self.ttc_fontpath])

        # Then
        self.assertEqual(len(fontlist), 0)
        self.assertEqual(m_ttfFontProperty.call_count, 1)

    @mock.patch("kiva.fonttools.font_manager.TTCollection",
                side_effect=RuntimeError)
    def test_ttc_exception_on_TTCollection(self, m_TTCollection):
        # When
        with self.assertLogs("kiva"):
            fontlist = createFontList([self.ttc_fontpath])

        # Then
        self.assertEqual(len(fontlist), 0)
        self.assertEqual(m_TTCollection.call_count, 1)


class TestTTFFontProperty(unittest.TestCase):
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
        entry = ttfFontProperty(test_font, TTFont(test_font))

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
        entry = ttfFontProperty(test_font, TTFont(test_font))

        # Then
        self.assertEqual(entry.name, exp_name)
        self.assertEqual(entry.style, exp_style)
        self.assertEqual(entry.variant, exp_variant)
        self.assertEqual(entry.weight, exp_weight)
        self.assertEqual(entry.stretch, exp_stretch)
        self.assertEqual(entry.size, exp_size)


class TestFontCache(unittest.TestCase):
    """ Test internal font cache building mechanism."""

    def setUp(self):
        self.ttf_files = [
            os.path.abspath(os.path.join(data_dir, "TestTTF.ttf"))
        ]

        temp_dir_obj = tempfile.TemporaryDirectory()
        self.temp_dir = temp_dir_obj.name
        self.addCleanup(temp_dir_obj.cleanup)

    def test_load_font_from_cache(self):
        # Test loading fonts from cache file.
        with patch_global_font_manager(None):
            with patch_font_cache(self.temp_dir, self.ttf_files):
                default_manager = font_manager_module.default_font_manager()

            # For some reasons, there are duplications in the list of files
            self.assertEqual(
                set(default_manager.ttffiles), set(self.ttf_files)
            )
            # The global singleton is now set.
            self.assertIsInstance(font_manager_module.fontManager, FontManager)

    def test_build_font_if_no_cache(self):
        # Calling default_font_manager will build the font cache
        # The temporary directory does not have a font cache file.
        with change_ets_app_dir(self.temp_dir) as cache_file:

            with patch_global_font_manager(None), \
                    patch_system_fonts(self.ttf_files):  # patch for speed
                font_manager_module.default_font_manager()

            # The cache file is created
            self.assertTrue(os.path.exists(cache_file))

    def test_no_import_side_effect(self):
        # Importing font_manager should have no side effect of creating
        # the font cache. Regression test for enthought/enable#362
        module_name = "kiva.fonttools.font_manager"
        modules = sys.modules
        original_module = modules.pop(module_name)
        with change_ets_app_dir(self.temp_dir) as cache_path:
            try:
                importlib.import_module(module_name)
            except Exception:
                raise
            else:
                # A cache is not created
                self.assertFalse(os.path.exists(cache_path))
            finally:
                modules[module_name] = original_module


class TestFontManager(unittest.TestCase):
    """ Test API of the font manager module."""

    def test_default_font_manager(self):
        font_manager = default_font_manager()
        self.assertIsInstance(font_manager, FontManager)


@contextlib.contextmanager
def change_ets_app_dir(dirpath):
    """ Temporarily change the application data directory in ETSConfig.

    Parameters
    ----------
    dirpath : str
        Path to be temporarily set to the ETSConfig.application_data
        so that it gets used for computing the font cache file path.

    Returns
    -------
    font_cache_file_path : str
        Path to the font cache in the given data directory.
        Returned for convenience.
    """
    original_data_dir = ETSConfig.application_data
    ETSConfig.application_data = dirpath
    try:
        yield font_manager_module._get_font_cache_path()
    finally:
        ETSConfig.application_data = original_data_dir


@contextlib.contextmanager
def patch_font_cache(dirpath, ttf_files):
    """ Patch the font cache content with the given list of FFT fonts
    and application data directory.

    Parameters
    ----------
    dirpath : str
        Path to be temporarily set to the ETSConfig.application_data
        so that it gets used for computing the font cache file path.
    ttf_files : list of str
        List of file paths to TTF files.

    Returns
    -------
    font_cache_file_path : str
        Path to the font cache in the given data directory.
        Returned for convenience.
    """
    with change_ets_app_dir(dirpath) as cache_file:
        with patch_system_fonts(ttf_files):
            font_manager_module._new_font_manager(cache_file)
        yield cache_file


def patch_system_fonts(ttf_files):
    """ Patch findSystemFonts with the given list of font file paths.

    This speeds up tests by avoiding having to parse a lot of font files
    on a system.

    Parameters
    ----------
    ttf_files : list of str
        List of file paths for TTF fonts
    """

    def fake_find_system_fonts(fontpaths=None, fontext="ttf"):
        if fontext == "ttf":
            return ttf_files
        return []

    return mock.patch(
        "kiva.fonttools.font_manager.findSystemFonts", fake_find_system_fonts
    )
