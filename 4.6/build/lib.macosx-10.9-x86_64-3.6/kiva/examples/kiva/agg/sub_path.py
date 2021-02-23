# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
import numpy

from kiva import agg

line_color = (0.0, 0.0, 0.0)
fill_color = numpy.array((200.0, 184.0, 106.0)) / 255.0
gc = agg.GraphicsContextArray((1600, 1600))
gc.rect(30, 30, 1200, 300)
gc.set_fill_color(fill_color)
gc.fill_path()

gc.set_fill_color((0.0, 0.0, 0.0, 0.4))
gc.translate_ctm(50, 50)
gc.move_to(10, 10)
gc.line_to(400, 400)

gc.move_to(410, 10)
gc.line_to(410, 400)
gc.line_to(710, 400)
gc.line_to(550, 300)
gc.line_to(710, 200)
gc.line_to(500, 10)
gc.close_path()

gc.rect(750, 10, 390, 390)
gc.draw_path()

gc.save("sub_path1.bmp")

line_color = (0.0, 0.0, 0.0)
fill_color = numpy.array((200.0, 184.0, 106.0)) / 255.0
gc = agg.GraphicsContextArray((1600, 1600))
gc.rect(30, 30, 1200, 300)
gc.set_fill_color(fill_color)
gc.fill_path()

gc.set_fill_color((0.0, 0.0, 0.0, 0.4))
gc.translate_ctm(50, 50)
gc.move_to(10, 10)
gc.line_to(400, 400)

gc.move_to(410, 10)
gc.line_to(410, 400)
gc.line_to(710, 400)
gc.curve_to(550, 300, 710, 200, 500, 10)
gc.close_path()

gc.rect(650, 10, 390, 390)
gc.fill_path()

gc.save("sub_path2.bmp")
