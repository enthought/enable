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

from traits.etsconfig.api import ETSConfig

from ._testing import patch_global_font_manager
from .. import font_manager as font_manager_module
from ..font_manager import default_font_manager, FontManager

data_dir = resource_filename("kiva.fonttools.tests", "data")


class TestFontCache(unittest.TestCase):
    """ Test internal font cache building mechanism."""

    def setUp(self):
        self.ttf_files = [
            os.path.abspath(os.path.join(data_dir, "TestTTF.ttf")),
            os.path.abspath(os.path.join(data_dir, "TestTTC.ttc")),
        ]

        temp_dir_obj = tempfile.TemporaryDirectory()
        self.temp_dir = temp_dir_obj.name
        self.addCleanup(temp_dir_obj.cleanup)

    def test_load_font_from_cache(self):
        # Test loading fonts from cache file.
        with patch_global_font_manager(None):
            with patch_font_cache(self.temp_dir, self.ttf_files):
                default_manager = font_manager_module.default_font_manager()

            # Check that all files are in the internal FontDatabase
            entries = default_manager.ttf_db.fonts_for_directory(data_dir)
            # Remove duplicates, since there may be more fonts than files.
            files = sorted(set(ent.fname for ent in entries))
            self.assertListEqual(files, sorted(self.ttf_files))

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
    """ Patch scan_system_fonts with the given list of font file paths.

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

    # Patch the version which was imported into `kiva.fonttools.font_manager`
    return mock.patch(
        "kiva.fonttools.font_manager.scan_system_fonts", fake_find_system_fonts
    )
