# (C) Copyright 2005-2022 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
from traits.api import HasStrictTraits
from traitsui.api import View, Item

from enable.trait_defs.api import KivaFont
from enable.trait_defs.ui.api import KivaFontEditor
from enable.examples._example_support import demo_main

from kiva.api import Font
from kiva.constants import ITALIC, SWISS, WEIGHT_BOLD


size = (500, 200)

sample_text = "Sphinx of black quartz, judge my vow."


class Demo(HasStrictTraits):
    """ An example which shows the KivaFontEditor's variations. """

    font = KivaFont(Font("Times", 24, SWISS, WEIGHT_BOLD, ITALIC))

    view = View(
        Item('font', style='simple', label="Simple"),
        Item('font', style='custom', label="Custom"),
        Item('font', style='text', label="Text"),
        Item('font', style='readonly', label="Readonly"),
        Item(
            'font',
            editor=KivaFontEditor(sample_text=sample_text),
            style='readonly',
            label="sample text",
        ),
        resizable=True,
        width=size[0],
        height=size[1],
    )


if __name__ == "__main__":
    # Save demo so that it doesn't get garbage collected when run within
    # existing event loop (i.e. from ipython).
    demo = demo_main(Demo, size=size)
