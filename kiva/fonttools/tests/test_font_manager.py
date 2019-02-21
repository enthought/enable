import os
import unittest
try:
    from unittest import mock
except ImportError:
    import mock

from pkg_resources import resource_filename

from ..font_manager import FontEntry, createFontList

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
