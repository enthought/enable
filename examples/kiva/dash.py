import time

import numpy

from kiva import constants
from kiva.agg import GraphicsContextArray


def dash(sz=(1000,1000)):
    gc = GraphicsContextArray(sz)
    gc.set_fill_color((1.0,0.0,0.0,0.1))
    gc.set_stroke_color((0.0,1.0,0.0,0.6))

    width = 10
    gc.set_line_width(10)

    phase = width * 2.5;
    pattern = width * numpy.array((5,5))
    gc.set_line_dash(pattern,phase)
    gc.set_line_cap(constants.CAP_BUTT)
    t1 = time.clock()
    gc.move_to(10,10)
    gc.line_to(sz[0]-10,sz[1]-10)
    gc.line_to(10,sz[1]-10)
    gc.close_path()
    gc.draw_path()
    t2 = time.clock()
    gc.save("dash.bmp")
    tot_time = t2 - t1
    print('time:', tot_time)


if __name__ == "__main__":
    dash()
