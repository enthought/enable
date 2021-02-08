# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
import os
import sys

from enable.savage.svg.document import SVGDocument
from enable.savage.trait_defs.ui.svg_editor import SVGEditor
from traits.api import HasTraits, Instance
from traitsui.api import Item, View

FILENAME = os.path.join(os.path.dirname(__file__), "lion.svg")


class StaticImageExample(HasTraits):
    svg = Instance(SVGDocument)

    traits_view = View(
        Item(
            "svg", editor=SVGEditor(), width=450, height=450, show_label=False
        ),
        resizable=True,
        title="StaticImageExample",
    )

    def __init__(self, filename, renderer, *args, **kw):
        super(StaticImageExample, self).__init__(*args, **kw)

        self.svg = SVGDocument.createFromFile(filename, renderer=renderer)


def main():
    if "--wx" in sys.argv:
        from enable.savage.svg.backends.wx.renderer import Renderer

        sys.argv.remove("--wx")
    elif "--kiva" in sys.argv:
        from enable.savage.svg.backends.kiva.renderer import Renderer

        sys.argv.remove("--kiva")
    else:
        from enable.savage.svg.backends.kiva.renderer import Renderer

    filename = sys.argv[1] if len(sys.argv) > 1 else FILENAME
    return StaticImageExample(filename, Renderer)


demo = main()

if __name__ == "__main__":
    demo.configure_traits()
