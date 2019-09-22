import os
import unittest
try:
    from unittest import mock
except ImportError:
    import mock

from pkg_resources import resource_filename
from fontTools.ttLib import TTFont

from ..font_manager import FontEntry, createFontList, ttfFontProperty

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
