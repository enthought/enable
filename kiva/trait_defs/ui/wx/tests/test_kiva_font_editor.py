""" Tests for ui.wx.kiva_font_editor
"""

import unittest

from kiva.tests._testing import is_wx, skip_if_not_wx

if is_wx():
    from kiva.trait_defs.ui.wx import kiva_font_editor


@skip_if_not_wx
class TestFacename(unittest.TestCase):

    def test_all_facenames(self):
        # Test loading of available face names does not fail.
        # The available face names depend on the system
        self.assertGreaterEqual(len(kiva_font_editor._all_facenames()), 0)
