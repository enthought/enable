from __future__ import print_function

import time
import sys

from kiva import agg

import tempfile

from enable.api import ConstraintsContainer, Component, ComponentEditor
from enable.primitives.image import Image

from traits.api import HasTraits, Instance
from traitsui.api import Item, View

if sys.platform == 'win32':
    now = time.clock
else:
    now = time.time

from lion_data import get_lion


def lion():
    sz = (1000, 1000)

    t1 = now()
    path_and_color, size, center = get_lion()
    t2 = now()
    print(t2 - t1)

    gc = agg.GraphicsContextArray(sz)
    t1 = now()

    gc.translate_ctm(sz[0] / 2., sz[1] / 2.)
    Nimages = 90
    for i in range(Nimages):
        for path, color in path_and_color:
            gc.begin_path()
            gc.add_path(path)
            gc.set_fill_color(color)
            gc.set_alpha(0.3)
            gc.fill_path()
        gc.rotate_ctm(1)
    t2 = now()
    print('total time, sec/image, img/sec:', t2 - t1, (t2 - t1) / Nimages,
          Nimages / (t2 - t1))
    file_path = tempfile.mktemp(suffix='.bmp')
    gc.save(file_path)
    return file_path


class Demo(HasTraits):
    canvas = Instance(Component)

    traits_view = View(Item('canvas', editor=ComponentEditor(),
                            show_label=False, width=200, height=200),
                       resizable=True)

    def _canvas_default(self):
        file_path = lion()
        image = Image.from_file(file_path, resist_width='weak',
                                resist_height='weak')

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
    main()

# EOF
