import os
import unittest
try:
    from unittest import mock
except:
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
