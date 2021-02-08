# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
from enable.api import Component
from enable.example_support import DemoFrame, demo_main


def glyph_a(gc):
    gc.move_to(28.47, 6.45)
    gc.quad_curve_to(21.58, 1.12, 19.82, 0.29)
    gc.quad_curve_to(17.19, -0.93, 14.21, -0.93)
    gc.quad_curve_to(9.57, -0.93, 6.57, 2.25)
    gc.quad_curve_to(3.56, 5.42, 3.56, 10.60)
    gc.quad_curve_to(3.56, 13.87, 5.03, 16.26)
    gc.quad_curve_to(7.03, 19.58, 11.99, 22.51)
    gc.quad_curve_to(16.94, 25.44, 28.47, 29.64)
    gc.line_to(28.47, 31.40)
    gc.quad_curve_to(28.47, 38.09, 26.34, 40.58)
    gc.quad_curve_to(24.22, 43.07, 20.17, 43.07)
    gc.quad_curve_to(17.09, 43.07, 15.28, 41.41)
    gc.quad_curve_to(13.43, 39.75, 13.43, 37.60)
    gc.line_to(13.53, 34.77)
    gc.quad_curve_to(13.53, 32.52, 12.38, 31.30)
    gc.quad_curve_to(11.23, 30.08, 9.38, 30.08)
    gc.quad_curve_to(7.57, 30.08, 6.42, 31.35)
    gc.quad_curve_to(5.27, 32.62, 5.27, 34.81)
    gc.quad_curve_to(5.27, 39.01, 9.57, 42.53)
    gc.quad_curve_to(13.87, 46.04, 21.63, 46.04)
    gc.quad_curve_to(27.59, 46.04, 31.40, 44.04)
    gc.quad_curve_to(34.28, 42.53, 35.64, 39.31)
    gc.quad_curve_to(36.52, 37.21, 36.52, 30.71)
    gc.line_to(36.52, 15.53)
    gc.quad_curve_to(36.52, 9.13, 36.77, 7.69)
    gc.quad_curve_to(37.01, 6.25, 37.57, 5.76)
    gc.quad_curve_to(38.13, 5.27, 38.87, 5.27)
    gc.quad_curve_to(39.65, 5.27, 40.23, 5.62)
    gc.quad_curve_to(41.26, 6.25, 44.19, 9.18)
    gc.line_to(44.19, 6.45)
    gc.quad_curve_to(38.72, -0.88, 33.74, -0.88)
    gc.quad_curve_to(31.35, -0.88, 29.93, 0.78)
    gc.quad_curve_to(28.52, 2.44, 28.47, 6.45)
    gc.close_path()

    gc.move_to(28.47, 9.62)
    gc.line_to(28.47, 26.66)
    gc.quad_curve_to(21.09, 23.73, 18.95, 22.51)
    gc.quad_curve_to(15.09, 20.36, 13.43, 18.02)
    gc.quad_curve_to(11.77, 15.67, 11.77, 12.89)
    gc.quad_curve_to(11.77, 9.38, 13.87, 7.06)
    gc.quad_curve_to(15.97, 4.74, 18.70, 4.74)
    gc.quad_curve_to(22.41, 4.74, 28.47, 9.62)
    gc.close_path()


class MyCanvas(Component):
    def draw(self, gc, **kwargs):
        w, h = gc.width(), gc.height()

        gc.move_to(0, 0)
        gc.line_to(w, h)
        gc.set_stroke_color((1, 0, 0))
        gc.stroke_path()
        gc.move_to(0, h)
        gc.line_to(w, 0)
        gc.set_stroke_color((0, 1, 0))
        gc.stroke_path()
        gc.rect(0, 0, w, h)
        gc.set_stroke_color((0, 0, 0, 0.5))
        gc.set_line_width(20)
        gc.stroke_path()

        gc.set_fill_color((0, 0, 1, 0.0))
        gc.rect(0, 0, w, h)

        gc.draw_path()

        gc.set_line_width(1)
        gc.translate_ctm(w / 2.0, h / 2.0)
        with gc:
            gc.scale_ctm(2.0, 2.0)
            glyph_a(gc)
        gc.stroke_path()

        gc.translate_ctm(0, -20)
        gc.scale_ctm(2.0, 2.0)
        glyph_a(gc)
        gc.set_fill_color((0, 0, 1, 1.0))
        gc.fill_path()


class Demo(DemoFrame):
    def _create_component(self):
        return MyCanvas()


if __name__ == "__main__":
    # Save demo so that it doesn't get garbage collected when run within
    # existing event loop (i.e. from ipython).
    demo = demo_main(Demo)
