# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
from time import perf_counter

from numpy import array

import kiva
from kiva import agg


def dash(sz=(1000, 1000)):
    gc = agg.GraphicsContextArray(sz)
    gc.set_fill_color((1.0, 0.0, 0.0, 0.1))
    gc.set_stroke_color((0.0, 1.0, 0.0, 0.6))

    width = 10
    gc.set_line_width(10)

    phase = width * 2.5
    pattern = width * array((5, 5))
    gc.set_line_dash(pattern, phase)
    gc.set_line_cap(kiva.CAP_BUTT)
    t1 = perf_counter()
    gc.move_to(10, 10)
    gc.line_to(sz[0] - 10, sz[1] - 10)
    gc.line_to(10, sz[1] - 10)
    gc.close_path()
    gc.draw_path()
    t2 = perf_counter()
    gc.save("dash.bmp")
    tot_time = t2 - t1
    print("time:", tot_time)


if __name__ == "__main__":
    dash()
