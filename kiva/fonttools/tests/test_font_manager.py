import glob
import importlib
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

from pkg_resources import resource_filename
from fontTools.ttLib import TTFont

from traits.etsconfig.api import ETSConfig

from ..font_manager import (
    createFontList,
    default_font_manager,
    findfont,
    FontEntry,
    FontProperties,
    FontManager,
    pickle_dump,
    pickle_load,
    ttfFontProperty,
)

data_dir = resource_filename('kiva.fonttools.tests', 'data')


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

    @mock.patch(
        "kiva.fonttools.font_manager.ttfFontProperty", side_effect=ValueError)
    def test_ttc_exception_on_ttfFontProperty(self, m_ttfFontProperty):
        # When
        fontlist = createFontList([self.ttc_fontpath])

        # Then
        self.assertEqual(len(fontlist), 0)
        self.assertEqual(m_ttfFontProperty.call_count, 1)

    @mock.patch(
        "kiva.fonttools.font_manager.TTCollection", side_effect=RuntimeError)
    def test_ttc_exception_on_TTCollection(self, m_TTCollection):
        # When
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

        self.temp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.temp_dir)

        self.cache_file = os.path.join(self.temp_dir, "font.cache")

        with patch_system_fonts(self.ttf_files):
            pickle_dump(FontManager(), self.cache_file)

    def test_load_font_cached_to_file(self):
        # patch import... fight import side-effect
        module_name = "kiva.fonttools.font_manager"
        modules = sys.modules
        original_data_dir = ETSConfig.application_data

        ETSConfig.application_data = self.temp_dir
        original_module = modules.pop(module_name)
        try:
            cache_dir = os.path.join(self.temp_dir, "kiva")
            os.makedirs(cache_dir)
            shutil.copyfile(
                self.cache_file,
                os.path.join(cache_dir, "fontList.cache")
            )
            new_module = importlib.import_module(module_name)
        except Exception:
            raise
        else:
            expected_manager = pickle_load(
                os.path.join(cache_dir, "fontList.cache")
            )
            self.assertEqual(
                new_module.default_font_manager().ttffiles,
                expected_manager.ttffiles,
            )
        finally:
            ETSConfig.application_data = original_data_dir
            modules[module_name] = original_module

    def test_rebuild_if_cache_not_found(self):
        module_name = "kiva.fonttools.font_manager"
        modules = sys.modules
        original_data_dir = ETSConfig.application_data

        ETSConfig.application_data = self.temp_dir
        original_module = modules.pop(module_name)
        try:
            new_module = importlib.import_module(module_name)
        except Exception:
            raise
        else:
            # A cache is created
            self.assertTrue(
                os.path.join(self.temp_dir, "kiva", "fontList.cache")
            )
        finally:
            ETSConfig.application_data = original_data_dir
            modules[module_name] = original_module


class TestFontManager(unittest.TestCase):
    """ Test API of the font manager module."""

    def test_default_font_manager(self):
        font_manager = default_font_manager()
        self.assertIsInstance(font_manager, FontManager)

    def test_findFont(self):
        # Warning because there are no families defined.
        with self.assertWarns(UserWarning):
            font = findfont(
                FontProperties(
                    family=[],
                    weight=500,
                )
            )
        # The returned value is a file path
        # This assumes there exists fonts on the system that can be loaded
        # by the font manager while the test is run.
        self.assertTrue(os.path.exists(font))


def patch_system_fonts(ttf_files):
    """ Patch findSystemFonts with the given list of font file paths.

    This speeds up tests by avoiding having to parse a lot of font files
    on a system.

    Parameters
    ----------
    ttf_files : list of str
        List of file paths for TTF fonts
    """

    def fake_find_system_fonts(fontpaths=None, fontext='ttf'):
        if fontext == "ttf":
            return ttf_files
        return []

    return mock.patch(
        "kiva.fonttools.font_manager.findSystemFonts",
        fake_find_system_fonts,
    )
