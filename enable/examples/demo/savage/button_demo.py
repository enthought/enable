# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
import os.path
from copy import copy

from traits.api import HasTraits, Str
from traitsui.api import HGroup, Item, View

from enable.savage.trait_defs.ui.svg_button import SVGButton

button_size = (64, 64)


class Demo(HasTraits):
    copy_button = SVGButton(
        label="Copy",
        filename=os.path.join(os.path.dirname(__file__), "edit-copy.svg"),
        width=button_size[0],
        height=button_size[1],
    )
    paste_button = SVGButton(
        label="Paste",
        filename=os.path.join(os.path.dirname(__file__), "edit-paste.svg"),
        width=button_size[0],
        height=button_size[1],
    )
    text = Str
    clipboard = Str

    traits_view = View(
        HGroup(
            Item("copy_button", show_label=False),
            Item(
                "paste_button",
                show_label=False,
                enabled_when="len(clipboard)>0",
            ),
        ),
        Item("text", width=200),
        title="SVG Button Demo",
    )

    def _copy_button_fired(self, event):
        self.clipboard = copy(self.text)

    def _paste_button_fired(self, event):
        self.text += self.clipboard


if __name__ == "__main__":
    demo = Demo()
    demo.configure_traits()
