# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
from numpy import cos, sin, arange, pi, array

from kiva.api import FILL, EOF_FILL, STROKE, FILL_STROKE, EOF_FILL_STROKE
from kiva.image import GraphicsContext


def draw_circle(gc, radius=2):
    gc.begin_path()
    angle = 0
    gc.move_to(radius * cos(angle), radius * sin(angle))
    for angle in arange(pi / 4, 2 * pi, pi / 4):
        gc.line_to(radius * cos(angle), radius * sin(angle))
    gc.close_path()
    gc.fill_path()


star_points = [(-20, -30), (0, 30), (20, -30), (-30, 10), (30, 10), (-20, -30)]
ring_point = (0, 35)
ring_radius = 5

fill_color = array((200.0, 184.0, 106.0)) / 255
point_color = array((0.3, 0.3, 0.3))
line_color = point_color

for i in range(len(star_points) + 1):
    gc = GraphicsContext((800, 800))
    gc.scale_ctm(8.0, 8.0)
    gc.translate_ctm(50, 50)

    # draw star
    gc.set_alpha(0.5)
    x, y = star_points[0]
    gc.move_to(x, y)
    for x, y in star_points[1:]:
        gc.line_to(x, y)
    gc.close_path()
    gc.set_fill_color(fill_color)
    gc.get_fill_color()
    gc.fill_path()

    gc.set_alpha(0.4)
    gc.set_stroke_color(line_color)
    gc.set_fill_color(line_color)
    gc.set_line_width(12)

    if i > 0:
        with gc:
            x, y = star_points[0]
            gc.translate_ctm(x, y)
            draw_circle(gc)

    if i > 1:
        points = star_points[:i]
        with gc:
            x, y = points[0]
            gc.move_to(x, y)
            for x, y in points[1:]:
                gc.line_to(x, y)
            gc.stroke_path()

    """
    for x,y in points:
        with gc:
            gc.translate_ctm(x,y)
            draw_circle(gc)
    """
    gc.save("star_path%d.bmp" % i)

# draw star
line_color = (0.0, 0.0, 0.0)
gc = GraphicsContext((800, 800))
gc.scale_ctm(8.0, 8.0)
gc.translate_ctm(50, 50)
gc.set_stroke_color(line_color)
gc.set_fill_color(fill_color)
gc.set_line_width(12)
x, y = star_points[0]
gc.move_to(x, y)
for x, y in star_points[1:]:
    gc.line_to(x, y)
gc.close_path()
gc.set_fill_color(fill_color)
gc.get_fill_color()
gc.draw_path()
gc.save("star_path7.bmp")

# draw star
gc = GraphicsContext((1700, 400))
line_color = (0.0, 0.0, 0.0)
gc.scale_ctm(4.0, 4.0)

offsets = array(((0, 0), (80, 0), (160, 0), (240, 0), (320, 0)))
modes = [FILL, EOF_FILL, STROKE, FILL_STROKE, EOF_FILL_STROKE]
pairs = list(zip(modes, offsets))
center = array((50, 50))
for mode, offset in pairs:
    with gc:
        xo, yo = center + offset
        gc.translate_ctm(xo, yo)
        gc.set_stroke_color(line_color)
        gc.set_fill_color(fill_color)
        gc.set_line_width(12)
        x, y = star_points[0]
        gc.move_to(x, y)
        for x, y in star_points[1:]:
            gc.line_to(x, y)
        gc.close_path()
        gc.set_fill_color(fill_color)
        gc.get_fill_color()
        gc.draw_path(mode)
gc.save("star_path8.bmp")
