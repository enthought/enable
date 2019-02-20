import os
import unittest

from ..font_manager import FontEntry, createFontList

HERE = os.path.dirname(__file__)

class TestCreateFontList(unittest.TestCase):

    def setUp(self):
        self.ttc_fontpath = os.path.join(HERE, "data", "TestTTC.ttc")

    def test_fontlist_from_ttc(self):
        # When
        fontlist = createFontList([self.ttc_fontpath])

        # Then
        self.assertEqual(len(fontlist), 2)
        for fontprop in fontlist:
            self.assertIsInstance(fontprop, FontEntry)
