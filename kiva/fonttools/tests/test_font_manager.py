import os
import unittest

from ..font_manager import FontEntry, createFontList

HERE = os.path.dirname(__file__)

class TestCreateFontList(unittest.TestCase):

    def test_fontlist_from_ttc(self):
        # Given
        fontpath = os.path.join(HERE, "data", "TestTTC.ttc")

        # When
        fontlist = createFontList([fontpath])

        # Then
        self.assertEqual(len(fontlist), 2)
        for fontprop in fontlist:
            self.assertIsInstance(fontprop, FontEntry)
