# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
import tempfile

from traits.api import Instance
from traitsui.api import Item, View

from enable.api import ConstraintsContainer, Component, ComponentEditor
from enable.example_support import demo_main, DemoFrame
from enable.primitives.image import Image
from kiva import constants
from kiva.image import GraphicsContext


def add_star(gc):
    gc.begin_path()

    # star
    gc.move_to(-20, -30)
    gc.line_to(0, 30)
    gc.line_to(20, -30)
    gc.line_to(-30, 10)
    gc.line_to(30, 10)
    gc.close_path()

    # line at top of star
    gc.move_to(-10, 30)
    gc.line_to(10, 30)


def stars():
    gc = GraphicsContext((500, 500))
    gc.translate_ctm(250, 300)
    add_star(gc)
    gc.draw_path()

    gc.translate_ctm(0, -100)
    add_star(gc)
    gc.set_fill_color((0.0, 0.0, 1.0))
    gc.draw_path(constants.EOF_FILL_STROKE)
    with tempfile.NamedTemporaryFile(suffix=".bmp") as fid:
        gc.save(fid.name)
        image = Image.from_file(
            fid.name, resist_width="weak", resist_height="weak"
        )
    return image


class Demo(DemoFrame):
    canvas = Instance(Component)

    traits_view = View(
        Item(
            "canvas",
            editor=ComponentEditor(),
            show_label=False,
            width=200,
            height=200,
        ),
        resizable=True,
    )

    def _canvas_default(self):
        image = stars()

        container = ConstraintsContainer(bounds=[500, 500])
        container.add(image)
        ratio = float(image.data.shape[1]) / image.data.shape[0]
        container.layout_constraints = [
            image.left == container.contents_left,
            image.right == container.contents_right,
            image.top == container.contents_top,
            image.bottom == container.contents_bottom,
            image.layout_width == ratio * image.layout_height,
        ]

        return container


if __name__ == "__main__":
    # Save demo so that it doesn't get garbage collected when run within
    # existing event loop (i.e. from ipython).
    demo = demo_main(Demo)
