# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
from kiva import agg
from kiva import constants

gc = agg.GraphicsContextArray((100, 100))
# gc.bmp_array[:5,:5] = (128,128,128,128)
gc.set_stroke_color((1, 0, 0))
# gc.move_to(0,0)
# gc.line_to(100,100)
# gc.stroke_path()

gc.set_fill_color((0, 0, 1))
# gc.rect(0,0,5,5)
gc.rect(0.5, 0.5, 5.0, 5.0)
gc.draw_path()
print(gc.bmp_array[:7, :7, 0])

gc.clear()
gc.set_line_cap(constants.CAP_SQUARE)
gc.set_line_join(constants.JOIN_MITER)
gc.set_fill_color((0, 0, 1))
# gc.rect(0,0,5,5)
gc.rect(0.5, 0.5, 5.0, 5.0)
gc.draw_path()
print(gc.bmp_array[:7, :7, 0])

# gc.save("pr.bmp")
