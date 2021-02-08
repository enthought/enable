# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
from enable.kiva_graphics_context import GraphicsContext
from kiva.api import Font, affine_from_rotation, invert

# Do some basic drawing tests and write the results out to files.
# This is mostly a python translation of the tests in kiva/agg/src/dummy.cpp


black = (0.0, 0.0, 0.0, 1.0)
white = (1.0, 1.0, 1.0, 1.0)
lightgray = (0.2, 0.2, 0.2, 1.0)
red = (1.0, 0.0, 0.0, 1.0)
green = (0.0, 1.0, 0.0, 1.0)
blue = (0.0, 0.0, 1.0, 1.0)
niceblue = (0.411, 0.584, 0.843, 1.0)

PI = 3.141_592_654


def draw_sub_image(gc, width, height):
    gc.clear(white)
    fill_color = green[:3] + (0.4,)  # We want green, but with an alpha of 0.4
    gc.set_fill_color(fill_color)
    gc.rect(0, 0, width, height)
    gc.fill_path()

    gc.set_stroke_color(red)
    gc.move_to(0.0, 0.0)
    gc.line_to(width, height)
    gc.stroke_path()
    gc.set_stroke_color(blue)
    gc.move_to(0.0, height)
    gc.line_to(width, 0.0)
    gc.stroke_path()


def test_arc_to2(gc, x2, y2, radiusstep=25.0):
    gc.set_stroke_color(lightgray)
    gc.move_to(0, 0)
    gc.line_to(100, 0)
    gc.line_to(x2, y2)
    gc.stroke_path()
    gc.set_stroke_color(black)

    numradii = 7
    for i in range(numradii):
        gc.move_to(0, 0)
        gc.arc_to(100, 0, x2, y2, i * radiusstep + 20.0)
    gc.stroke_path()


def test_arc_curve(gc):
    with gc:
        gc.translate_ctm(50.0, 50.0)
        gc.rotate_ctm(PI / 8)
        gc.set_stroke_color(blue)
        gc.rect(0.5, 0.5, 210, 210)
        gc.stroke_path()
        gc.set_stroke_color(black)
        gc.set_line_width(1)
        gc.move_to(50.5, 25.5)
        gc.arc(50.5, 50.5, 50.0, 0.0, PI / 2, False)
        gc.move_to(100.5, 50.5)
        gc.arc(100.5, 50.5, 50.0, 0.0, -PI / 2 * 0.8, False)
        gc.stroke_path()

    with gc:
        gc.translate_ctm(250.5, 50.5)
        gc.set_stroke_color(blue)
        gc.rect(0.5, 0.5, 250.0, 250.0)
        gc.stroke_path()
        gc.set_stroke_color(red)
        gc.move_to(100.0, 100.0)
        gc.line_to(100.0, 150.0)
        gc.arc_to(100.0, 200.0, 150.0, 200.0, 50.0)
        gc.line_to(200.0, 200.0)
        gc.close_path()
        gc.stroke_path()


def test_arc_to(gc):
    # We don't have compiled paths yet, so we simulate them by python functions
    def axes(gc):
        gc.move_to(0.5, 50.5)
        gc.line_to(100.5, 50.5)
        gc.move_to(50.5, 0.5)
        gc.line_to(50.5, 100.5)

    def box(gc):
        gc.move_to(0.5, 0.5)
        gc.line_to(100.5, 0.5)
        gc.line_to(100.5, 100.5)
        gc.line_to(0.5, 100.5)
        gc.close_path()

    def arc(gc):
        gc.move_to(10, 10)
        gc.line_to(20, 10)
        gc.arc_to(40, 10, 40, 30, 20)
        gc.line_to(40, 40)

    def whole_shebang(gc):
        with gc:
            axes(gc)
            box(gc)
            gc.translate_ctm(0.0, 50.5)
            arc(gc)

            gc.translate_ctm(50.5, 50.5)
            gc.rotate_ctm(-PI / 2)
            arc(gc)
            gc.rotate_ctm(PI / 2)

            gc.translate_ctm(50.5, -50.5)
            gc.rotate_ctm(-PI)
            arc(gc)
            gc.rotate_ctm(PI)

            gc.translate_ctm(-50.5, -50.5)
            gc.rotate_ctm(-3 * PI / 2)
            arc(gc)

    gc.set_stroke_color(red)
    gc.set_line_width(1.0)
    with gc:
        gc.translate_ctm(50.5, 300.5)
        whole_shebang(gc)
        gc.stroke_path()

        gc.translate_ctm(130.5, 50.0)
        with gc:
            gc.rotate_ctm(PI / 6)
            whole_shebang(gc)
            gc.set_stroke_color(blue)
            gc.stroke_path()

        gc.translate_ctm(130.5, 0.0)
        with gc:
            gc.rotate_ctm(PI / 3)
            gc.scale_ctm(1.0, 2.0)
            whole_shebang(gc)
            gc.stroke_path()

    with gc:
        gc.translate_ctm(150.5, 20.5)
        test_arc_to2(gc, 160.4, 76.5, 50.0)

    gc.translate_ctm(120.5, 100.5)
    gc.scale_ctm(-1.0, 1.0)
    test_arc_to2(gc, 70.5, 96.5)
    gc.translate_ctm(-300.5, 100.5)
    gc.scale_ctm(0.75, -1.0)
    test_arc_to2(gc, 160.5, 76.5, 50.0)


def test_simple_clip_stack(gc):
    gc.clear(white)
    gc.clip_to_rect(100.0, 100.0, 1.0, 1.0)
    gc.rect(0.0, 0.0, gc.width(), gc.height())
    gc.set_fill_color(red)
    gc.fill_path()


def test_clip_stack(gc):
    sub_windows = (
        (10.5, 250, 200, 200),
        (220.5, 250, 200, 200),
        (430.5, 250, 200, 200),
        (10.5, 10, 200, 200),
        (220.5, 10, 200, 200),
        (430.5, 10, 200, 200),
    )
    gc.set_line_width(2)
    gc.set_stroke_color(black)
    gc.rects(sub_windows)
    gc.stroke_path()

    img = GraphicsContext((200, 200))

    main_rects = ((40.5, 30.5, 120, 50), (40.5, 120.5, 120, 50))
    disjoint_rects = ((60.5, 115.5, 80, 15), (60.5, 70.5, 80, 15))
    vert_rect = (60.5, 10.5, 55, 180)

    # Draw the full image
    draw_sub_image(img, 200, 200)
    gc.draw_image(img, sub_windows[0])
    img.clear()

    # First clip
    img.clip_to_rects(main_rects)
    draw_sub_image(img, 200, 200)
    gc.draw_image(img, sub_windows[1])

    # Second Clip
    with img:
        img.clear()
        img.clip_to_rects(main_rects)
        img.clip_to_rect(*vert_rect)
        draw_sub_image(img, 200, 200)
        gc.draw_image(img, sub_windows[2])

    # Pop back to first clip
    img.clear()
    draw_sub_image(img, 200, 200)
    gc.draw_image(img, sub_windows[3])

    # Adding a disjoing set of rects
    img.clear()
    with img:
        img.clip_to_rects(main_rects)
        img.clip_to_rects(disjoint_rects)
        draw_sub_image(img, 200, 200)
        gc.draw_image(img, sub_windows[4])

    # Pop back to first clip
    img.clear()
    draw_sub_image(img, 200, 200)
    gc.draw_image(img, sub_windows[5])


def test_handling_text(gc):
    font = Font(face_name="Arial", size=32)
    gc.set_font(font)
    gc.translate_ctm(100.0, 100.0)
    gc.move_to(-5, 0)
    gc.line_to(5, 0)
    gc.move_to(0, 5)
    gc.line_to(0, -5)
    gc.move_to(0, 0)
    gc.stroke_path()
    txtRot = affine_from_rotation(PI / 6)
    gc.set_text_matrix(txtRot)
    gc.show_text("Hello")
    txtRot = invert(txtRot)
    gc.set_text_matrix(txtRot)
    gc.show_text("inverted")


if __name__ == "__main__":
    from kiva import ps, svg

    tests = (
        test_clip_stack,
        #        test_arc_to, # not supported
        test_handling_text,
    )
    gcs = {"eps": ps.PSGC, "svg": svg.GraphicsContext}

    for fmt, gc in gcs.items():
        for test_func in tests:
            context = gc((800, 600))
            test_func(context)
            context.save(test_func.__name__ + "." + fmt)
