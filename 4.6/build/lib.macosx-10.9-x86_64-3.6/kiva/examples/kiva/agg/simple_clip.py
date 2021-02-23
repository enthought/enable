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

from kiva import agg

samples = 1
pt = (250, 250)
sz = (100, 100)

transparent = (1, 1, 1, 0.0)
gc_img = agg.GraphicsContextArray((500, 500))
# gc_img.clear(transparent)
# clear isn't working...
gc_img.bmp_array[:] = (255, 255, 255, 0)
gc_img.set_fill_color((1, 0, 0, 0.5))
gc_img.set_stroke_color((0, 0, 1))
gc_img.rect(pt[0], pt[1], sz[0], sz[1])
gc_img.draw_path()

gc_main = agg.GraphicsContextArray((500, 500))
gc_main.set_fill_color((0, 0, 1, 0.5))
gc_main.set_stroke_color((0, 1, 0))
gc_main.rect(300, 300, 100, 100)
gc_main.draw_path()

gc_main.clip_to_rect(pt[0], pt[1], sz[0], sz[1])
t1 = perf_counter()
for i in range(samples):
    gc_main.draw_image(gc_img)
t2 = perf_counter()
print("with clip", t2 - t1)

gc_main.save("with_clip.bmp")

# gc_main.clear(transparent)
gc_main.bmp_array[:] = (255, 255, 255, 255)
gc_main.clear_clip_path()
gc_main.set_fill_color((0, 0, 1, 0.5))
gc_main.set_stroke_color((0, 1, 0))
gc_main.rect(300, 300, 100, 100)
gc_main.draw_path()

t1 = perf_counter()
for i in range(samples):
    gc_main.draw_image(gc_img)
t2 = perf_counter()
print("without clip", t2 - t1)

gc_main.save("without_clip.bmp")
