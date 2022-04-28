# (C) Copyright 2005-2022 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
""" Test the interaction between traitsui and enable's ComponentEditor.
"""
import unittest

from traits.api import HasTraits
from traits.testing.api import UnittestTools
from traitsui.api import Item, View

from kiva.api import Font
from enable.enable_traits import font_trait
from enable.trait_defs.ui.kiva_font_editor import KivaFontEditor
from enable.tests._testing import skip_if_null


ITEM_WIDTH, ITEM_HEIGHT = 700, 200


class KivaFontView(HasTraits):
    """ View containing an item with ComponentEditor. """

    font = font_trait()

    traits_view = View(
        Item("font", editor=KivaFontEditor()),
        resizable=True,
    )


class TestKivaFontEditor(UnittestTools, unittest.TestCase):

    @skip_if_null
    def test_readonly_default_view(self):
        obj = KivaFontView()

        ui = obj.edit_traits(
            view=View(
                Item("font", editor=KivaFontEditor(), style='readonly'),
                resizable=True,
            )
        )
        try:
            # check initial state
            editor = ui.info.font
            component = editor.component
            self.assertIs(editor.value, obj.font)
            self.assertIs(editor.font, obj.font)
            self.assertIs(component.font, obj.font)
            self.assertEqual(editor.str_value, "10 point")
            self.assertEqual(component.text, "10 point")

            # check a change
            new_font = Font(
                face_name="Helvetica",
                size=24,
                weight=700,
                style=2,
            )

            obj.font = new_font

            self.assertIs(editor.value, new_font)
            self.assertIs(component.font, new_font)
            self.assertEqual(
                editor.str_value, "24 point Helvetica Bold Italic")
            self.assertEqual(component.text, "24 point Helvetica Bold Italic")
        finally:
            ui.dispose()

    @skip_if_null
    def test_simple_default_view(self):
        obj = KivaFontView()

        ui = obj.edit_traits(
            view=View(
                Item("font", editor=KivaFontEditor()),
                resizable=True,
            )
        )
        try:
            editor = ui.info.font
            component = editor.component
            self.assertIs(editor.value, obj.font)
            self.assertIs(editor.font, obj.font)
            self.assertIs(component.font, obj.font)
            self.assertEqual(editor.str_value, "10 point")
            self.assertEqual(component.text, "10 point")
        finally:
            ui.dispose()

    @skip_if_null
    def test_simple_default_object_change(self):
        obj = KivaFontView()

        ui = obj.edit_traits(
            view=View(
                Item("font", editor=KivaFontEditor()),
                resizable=True,
            )
        )
        try:
            editor = ui.info.font
            component = editor.component

            new_font = Font(
                face_name="Helvetica",
                size=24,
                weight=700,
                style=2,
            )
            obj.font = new_font

            self.assertIs(editor.value, new_font)
            self.assertIs(editor.font, new_font)
            self.assertIs(component.font, new_font)
            self.assertEqual(
                editor.str_value, "24 point Helvetica Bold Italic")
            self.assertEqual(component.text, "24 point Helvetica Bold Italic")
        finally:
            ui.dispose()

    @skip_if_null
    def test_simple_default_editor_change(self):
        obj = KivaFontView()

        ui = obj.edit_traits(
            view=View(
                Item("font", editor=KivaFontEditor()),
                resizable=True,
            )
        )
        try:
            editor = ui.info.font
            component = editor.component

            new_font = Font(
                face_name="Helvetica",
                size=24,
                weight=700,
                style=2,
            )
            editor.update_object(new_font)

            self.assertIs(obj.font, new_font)
            self.assertIs(editor.value, new_font)
            self.assertIs(editor.font, new_font)
            self.assertIs(component.font, new_font)
            self.assertEqual(
                editor.str_value, "24 point Helvetica Bold Italic")
            self.assertEqual(component.text, "24 point Helvetica Bold Italic")
        finally:
            ui.dispose()

    @skip_if_null
    def test_sample_text(self):
        # this is a smoke test
        obj = KivaFontView()

        ui = obj.edit_traits(
            view=View(
                Item(
                    "font",
                    editor=KivaFontEditor(sample_text="sample text"),
                    style='readonly',
                ),
                resizable=True,
            )
        )
        try:
            editor = ui.info.font
            component = editor.component
            self.assertEqual(editor.str_value, "sample text")
            self.assertEqual(component.text, "sample text")
        finally:
            ui.dispose()
