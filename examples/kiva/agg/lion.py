from __future__ import print_function

import time
import sys

from kiva import agg
from lion_data import get_lion

try:
    from time import perf_counter
except ImportError:
    from time import clock as perf_counter


def main():
    sz = (1000, 1000)

    t1 = perf_counter()
    path_and_color, _size, _center = get_lion()
    t2 = perf_counter()
    print(t2 - t1)

    gc = agg.GraphicsContextArray(sz)
    t1 = perf_counter()

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
    t2 = perf_counter()
    print('total time, sec/image, img/sec:', t2 - t1, (t2 - t1) / Nimages, Nimages / (t2 - t1))
    gc.save('lion.bmp')


if __name__ == "__main__":
    main()

# EOF
