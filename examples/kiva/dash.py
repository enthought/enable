from __future__ import print_function

import tempfile
try:
    from time import perf_counter
except ImportError:
    from time import clock as perf_counter

import numpy

from enable.api import ConstraintsContainer
from enable.example_support import DemoFrame, demo_main
from enable.primitives.image import Image
from kiva import constants
from kiva.agg import GraphicsContextArray


def dash(sz=(1000, 1000)):
    gc = GraphicsContextArray(sz)
    gc.set_fill_color((1.0, 0.0, 0.0, 0.1))
    gc.set_stroke_color((0.0, 1.0, 0.0, 0.6))

    width = 10
    gc.set_line_width(10)

    phase = width * 2.5
    pattern = width * numpy.array((5, 5))
    gc.set_line_dash(pattern, phase)
    gc.set_line_cap(constants.CAP_BUTT)
    t1 = perf_counter()
    gc.move_to(10, 10)
    gc.line_to(sz[0] - 10, sz[1] - 10)
    gc.line_to(10, sz[1] - 10)
    gc.close_path()
    gc.draw_path()
    t2 = perf_counter()
    file_path = tempfile.mktemp(suffix='.bmp')
    gc.save(file_path)
    tot_time = t2 - t1
    print('time:', tot_time)
    return file_path


class Demo(DemoFrame):

    def _create_component(self):
        file_path = dash()
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
    # Save demo so that it doesn't get garbage collected when run within
    # existing event loop (i.e. from ipython).
    demo = demo_main(Demo)
