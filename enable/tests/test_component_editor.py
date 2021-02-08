# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
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

from enable.component_editor import ComponentEditor
from traits.api import Any, HasTraits
from traitsui.api import Item, View

from enable.tests._testing import get_dialog_size, skip_if_null

ITEM_WIDTH, ITEM_HEIGHT = 700, 200


class _ComponentDialog(HasTraits):
    """ View containing an item with ComponentEditor. """

    thing = Any

    traits_view = View(
        Item("thing", editor=ComponentEditor(), show_label=False),
        resizable=True,
    )


class _ComponentDialogWithSize(HasTraits):
    """ View containing an item with ComponentEditor and given size. """

    thing = Any

    traits_view = View(
        Item(
            "thing",
            editor=ComponentEditor(),
            show_label=False,
            width=ITEM_WIDTH,
            height=ITEM_HEIGHT,
        ),
        resizable=True,
    )


class TestComponentEditor(unittest.TestCase):
    @skip_if_null
    def test_initial_component(self):
        # BUG: the initial size of an Item with ComponentEditor is zero
        # in the Qt backend

        dialog = _ComponentDialog()
        ui = dialog.edit_traits()

        size = get_dialog_size(ui.control)
        self.assertGreater(size[0], 0)
        self.assertGreater(size[1], 0)

    @skip_if_null
    def test_initial_component_with_item_size(self):
        # BEH: the initial component size should respect the size of the

        dialog = _ComponentDialogWithSize()
        ui = dialog.edit_traits()

        size = get_dialog_size(ui.control)

        self.assertGreater(size[0], ITEM_WIDTH - 1)
        self.assertGreater(size[1], ITEM_HEIGHT - 1)
