# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
import kiva
from kiva import agg


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


gc = agg.GraphicsContextArray((500, 500))
gc.translate_ctm(250, 300)
add_star(gc)
gc.draw_path()

gc.translate_ctm(0, -100)
add_star(gc)
gc.set_fill_color((0.0, 0.0, 1.0))
gc.draw_path(kiva.EOF_FILL_STROKE)

gc.save("star1.bmp")
