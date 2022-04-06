# (C) Copyright 2005-2022 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
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
        editor_factory = kiva_font_editor.KivaFontEditor()
        self.assertGreaterEqual(len(editor_factory.all_facenames()), 0)
