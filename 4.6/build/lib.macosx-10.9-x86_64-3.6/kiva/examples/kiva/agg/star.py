# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
from math import pi

from kiva import agg


def add_star(gc):
    gc.begin_path()
    gc.move_to(-20, -30)
    gc.line_to(0, 30)
    gc.line_to(20, -30)
    gc.line_to(-30, 10)
    gc.line_to(30, 10)
    gc.close_path()
    gc.move_to(-10, 30)
    gc.line_to(10, 30)


gc = agg.GraphicsContextArray((500, 500))

with gc:
    gc.set_alpha(0.3)
    gc.set_stroke_color((1.0, 0.0, 0.0))
    gc.set_fill_color((0.0, 1.0, 0.0))

    for i in range(0, 600, 5):
        with gc:
            gc.translate_ctm(i, i)
            gc.rotate_ctm(i * pi / 180.0)
            add_star(gc)
            gc.draw_path()

gc.set_fill_color((0.5, 0.5, 0.5, 0.4))
gc.rect(150, 150, 200, 200)
gc.fill_path()
gc.save("star.bmp")
