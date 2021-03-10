# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!

# ****************************************
# * READ THIS BEFORE ADDING NEW IMPORTS: *
# ****************************************
#
# The benchmark runner in .bench (see the `gen_suite` function) uses the
# `inspect.isclass` to automatically discover the available benchmarks in
# this module. Therefore one must be careful to import only module objects
# or functions only. Any classes will get picked up as benchmarks.

import math

import numpy as np

import kiva.api as kiva_api


def gen_image():
    img = np.zeros((512, 512, 4), dtype=np.uint8)
    img[100:300, 100:300, (0, 3)] = 255
    img[10:400, 10:400, (1, 3)] = 255
    img[195:400, 195:400, (2, 3)] = 255
    img[410:500, 410:500, :] = 255
    return img


def gen_large_path(obj):
    obj.arc(250, 250, 240, 0.0, 2 * math.pi)
    for x in range(50, 500, 50):
        obj.move_to(x, 0)
        obj.line_to(x, 500)
    for y in range(50, 500, 50):
        obj.move_to(0, y)
        obj.line_to(500, y)

    return obj


def gen_small_path(obj):
    for x in range(5, 25, 5):
        obj.move_to(x, 5)
        obj.line_to(x, 20)
    for y in range(5, 25, 5):
        obj.move_to(5, y)
        obj.line_to(20, y)

    return obj


def gen_points(count=100):
    points = (np.random.random(size=count) * 500.0)
    return points.reshape(count // 2, 2)


def gen_moderate_complexity_path(obj):
    obj.arc(300, 200, 100, math.pi, 1.5*math.pi)
    obj.move_to(300, 100)
    obj.line_to(500, 150)
    obj.line_to(300, 200)
    obj.curve_to(230, 150, 270, 250, 200, 200)

    return obj


class draw_path:
    def __init__(self, gc, module):
        self.gc = gc

    def __call__(self):
        with self.gc:
            gen_large_path(self.gc)
            self.gc.set_stroke_color((0.33, 0.66, 0.99, 1.0))
            self.gc.set_line_width(5.0)
            self.gc.stroke_path()


class draw_rect:
    def __init__(self, gc, module):
        self.points = gen_points(200)
        self.gc = gc

    def __call__(self):
        with self.gc:
            self.gc.set_fill_color((0.33, 0.66, 0.99, 1.0))
            for pt in self.points:
                self.gc.rect(pt[0], pt[1], 15.0, 15.0)
            self.gc.fill_path()


class draw_marker_at_points:
    def __init__(self, gc, module):
        self.points = gen_points(1000)
        self.gc = gc

    def __call__(self):
        self.gc.draw_marker_at_points(
            self.points, 5.0, kiva_api.PLUS_MARKER
        )


class draw_path_at_points:
    def __init__(self, gc, module):
        self.points = gen_points()
        self.path = gen_small_path(getattr(module, 'CompiledPath')())
        self.gc = gc

    def __call__(self):
        with self.gc:
            self.gc.set_stroke_color((0.99, 0.66, 0.33, 0.75))
            self.gc.set_line_width(1.5)
            self.gc.draw_path_at_points(
                self.points, self.path, kiva_api.STROKE
            )


class draw_image:
    def __init__(self, gc, module):
        self.img = gen_image()
        self.gc = gc

    def __call__(self):
        self.gc.draw_image(self.img, (0, 0, 512, 512))


class show_text:
    def __init__(self, gc, module):
        self.text = [
            'The quick brown',
            'fox jumped over',
            'the lazy dog',
            '狐假虎威',
        ]
        self.font = kiva_api.Font('Times New Roman', size=72)
        self.gc = gc

    def __call__(self):
        with self.gc:
            self.gc.set_fill_color((0.5, 0.5, 0.0, 1.0))
            self.gc.set_font(self.font)
            y = 512 - self.font.size * 1.4
            for line in self.text:
                self.gc.set_text_position(4, y)
                self.gc.show_text(line)
                y -= self.font.size * 1.4


class draw_path_linear_gradient:
    def __init__(self, gc, module):
        # colors are 5 doubles: offset, red, green, blue, alpha
        starting_color = [0.0, 1.0, 1.0, 1.0, 1.0]
        ending_color = [1.0, 0.0, 0.0, 0.0, 1.0]
        self.gradient = np.array([starting_color, ending_color])
        self.gc = gc

    def __call__(self):
        with self.gc:
            gen_moderate_complexity_path(self.gc)
            self.gc.linear_gradient(
                200, 200, 500, 150,
                self.gradient,
                'pad',
            )
            self.gc.fill_path()


class show_text_radial_gradient:
    def __init__(self, gc, module):
        starting_color = [0.0, 1.0, 1.0, 1.0, 1.0]
        mid_color = [0.5, 1.0, 0.0, 1.0, 1.0]
        ending_color = [1.0, 1.0, 1.0, 0.0, 1.0]
        self.gradient = np.array([starting_color, mid_color, ending_color])
        self.text = [
            'The quick brown',
            'fox jumped over',
            'the lazy dog',
            '狐假虎威',
        ]
        self.font = kiva_api.Font('Times New Roman', size=72)
        self.gc = gc

    def __call__(self):
        with self.gc:
            self.gc.radial_gradient(
                256, 256, 300, 256, 256,
                self.gradient,
                'pad',
            )
            self.gc.set_text_drawing_mode(kiva_api.TEXT_FILL_STROKE)
            self.gc.set_line_width(0.5)
            self.gc.set_font(self.font)
            y = 512 - self.font.size * 1.4
            for line in self.text:
                self.gc.set_text_position(4, y)
                self.gc.show_text(line)
                y -= self.font.size * 1.4
