# (C) Copyright 2005-2023 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
import contextlib
import os
import shutil
import tempfile

import numpy
from PIL import Image

from kiva.api import (
    DECORATIVE, DEFAULT, ITALIC, MODERN, NORMAL, ROMAN, SCRIPT, TELETYPE, Font
)
from kiva.constants import (
    FILL_STROKE, WEIGHT_THIN, WEIGHT_EXTRALIGHT, WEIGHT_LIGHT, WEIGHT_NORMAL,
    WEIGHT_MEDIUM, WEIGHT_SEMIBOLD, WEIGHT_BOLD, WEIGHT_EXTRABOLD,
    WEIGHT_HEAVY, WEIGHT_EXTRAHEAVY
)


families = [DECORATIVE, DEFAULT, MODERN, ROMAN, SCRIPT, TELETYPE]

weights = [
    WEIGHT_THIN, WEIGHT_EXTRALIGHT, WEIGHT_LIGHT, WEIGHT_NORMAL, WEIGHT_MEDIUM,
    WEIGHT_SEMIBOLD, WEIGHT_BOLD, WEIGHT_EXTRABOLD, WEIGHT_HEAVY,
    WEIGHT_EXTRAHEAVY,
]


rgba_float_dtype = numpy.dtype([
    ('red', "float64"),
    ('green', "float64"),
    ('blue', "float64"),
    ('alpha', "float64"),
])
rgb_float_dtype = numpy.dtype([
    ('red', "float64"),
    ('green', "float64"),
    ('blue', "float64"),
])


class DrawingTester(object):
    """ Basic drawing tests for graphics contexts.
    """

    def setUp(self):
        self.directory = tempfile.mkdtemp()
        self.filename = os.path.join(self.directory, "rendered")
        self.gc = self.create_graphics_context(300, 300)
        self.gc.clear()
        self.gc.set_stroke_color((1.0, 0.0, 0.0))
        self.gc.set_fill_color((1.0, 0.0, 0.0))
        self.gc.set_line_width(5)

    def tearDown(self):
        del self.gc
        shutil.rmtree(self.directory)

    def test_image(self):
        img = numpy.zeros((20, 20, 4), dtype=numpy.uint8)
        img[5:15, 5:15, (0, 3)] = 255
        with self.draw_and_check():
            self.gc.draw_image(img, (100, 100, 20, 20))

    def test_line(self):
        with self.draw_and_check():
            self.gc.begin_path()
            self.gc.move_to(107, 204)
            self.gc.line_to(107, 104)
            self.gc.stroke_path()

    def test_rectangle(self):
        with self.draw_and_check():
            self.gc.begin_path()
            self.gc.move_to(107, 104)
            self.gc.line_to(107, 184)
            self.gc.line_to(187, 184)
            self.gc.line_to(187, 104)
            self.gc.line_to(107, 104)
            self.gc.stroke_path()

    def test_rect(self):
        with self.draw_and_check():
            self.gc.begin_path()
            self.gc.rect(0, 0, 200, 200)
            self.gc.stroke_path()

    def test_circle(self):
        with self.draw_and_check():
            self.gc.begin_path()
            self.gc.arc(150, 150, 100, 0.0, 2 * numpy.pi)
            self.gc.stroke_path()

    def test_quarter_circle(self):
        with self.draw_and_check():
            self.gc.begin_path()
            self.gc.arc(150, 150, 100, 0.0, numpy.pi / 2)
            self.gc.stroke_path()

    def test_arc_to(self):
        with self.draw_and_check():
            self.gc.begin_path()
            self.gc.move_to(0, 50)
            self.gc.arc_to(0, 150, 100, 150, 50)
            self.gc.stroke_path()

    def test_quad_curve_to(self):
        with self.draw_and_check():
            self.gc.begin_path()
            self.gc.move_to(0, 50)
            self.gc.quad_curve_to(0, 100, 50, 100)
            self.gc.stroke_path()

    def test_curve_to(self):
        with self.draw_and_check():
            self.gc.begin_path()
            self.gc.move_to(300, 100)
            self.gc.curve_to(230, 150, 270, 250, 200, 200)
            self.gc.stroke_path()

    def test_text(self):
        for family in families:
            for weight in weights:
                for style in [NORMAL, ITALIC]:
                    with self.subTest(family=family, weight=weight, style=style):
                        self.gc = self.create_graphics_context()
                        with self.draw_and_check():
                            font = Font(family=family)
                            font.size = 24
                            font.weight = weight
                            font.style = style
                            self.gc.set_font(font)
                            self.gc.set_text_position(23, 67)
                            self.gc.show_text("hello kiva")

    def test_circle_fill(self):
        with self.draw_and_check():
            self.gc.begin_path()
            self.gc.arc(150, 150, 100, 0.0, 2 * numpy.pi)
            self.gc.fill_path()

    def test_star_fill(self):
        with self.draw_and_check():
            self.gc.begin_path()
            self.gc.move_to(100, 100)
            self.gc.line_to(150, 200)
            self.gc.line_to(200, 100)
            self.gc.line_to(100, 150)
            self.gc.line_to(200, 150)
            self.gc.line_to(100, 100)
            self.gc.fill_path()

    def test_star_eof_fill(self):
        with self.draw_and_check():
            self.gc.begin_path()
            self.gc.move_to(100, 100)
            self.gc.line_to(150, 200)
            self.gc.line_to(200, 100)
            self.gc.line_to(100, 150)
            self.gc.line_to(200, 150)
            self.gc.line_to(100, 100)
            self.gc.eof_fill_path()

    def test_circle_clip(self):
        with self.draw_and_check():
            self.gc.clip_to_rect(150, 150, 100, 100)
            self.gc.begin_path()
            self.gc.arc(150, 150, 100, 0.0, 2 * numpy.pi)
            self.gc.fill_path()

    def test_text_clip(self):
        with self.draw_and_check():
            self.gc.clip_to_rect(23, 77, 100, 23)
            font = Font(family=MODERN)
            font.size = 24
            self.gc.set_font(font)
            self.gc.set_text_position(23, 67)
            self.gc.show_text("hello kiva")

    def test_star_clip(self):
        with self.draw_and_check():
            self.gc.begin_path()
            self.gc.move_to(100, 100)
            self.gc.line_to(150, 200)
            self.gc.line_to(200, 100)
            self.gc.line_to(100, 150)
            self.gc.line_to(200, 150)
            self.gc.line_to(100, 100)
            self.gc.close_path()
            self.gc.clip()

            self.gc.begin_path()
            self.gc.arc(150, 150, 100, 0.0, 2 * numpy.pi)
            self.gc.fill_path()

    def test_draw_path_at_points(self):
        if not hasattr(self.gc, 'draw_path_at_points'):
            self.skipTest("GC doesn't have 'draw_marker_at_points' method.")

        path = self.gc.get_empty_path()
        path.move_to(-5, -5)
        path.line_to(-5, 5)
        path.line_to(5, 5)
        path.line_to(5, -5)
        path.close_path()

        points = numpy.array([[0, 0], [10, 10], [20, 20], [30, 30]])

        with self.draw_and_check():
            self.gc.draw_path_at_points(points, path, FILL_STROKE)
            self.gc.fill_path()

    def test_set_stroke_color(self):
        # smoke tests for different color types that should be accepted
        colors = [
            (0.4, 0.2, 0.6),
            [0.4, 0.2, 0.6],
            numpy.array([0.4, 0.2, 0.6]),
            numpy.array([(0.4, 0.2, 0.6)], dtype=rgb_float_dtype)[0],
            (0.4, 0.2, 0.6, 1.0),
            [0.4, 0.2, 0.6, 1.0],
            numpy.array([0.4, 0.2, 0.6, 1.0]),
            numpy.array([(0.4, 0.2, 0.6, 1.0)], dtype=rgba_float_dtype)[0],
        ]
        for color in colors:
            with self.subTest(color=color):
                with self.gc:
                    self.gc.set_stroke_color(color)

    def test_set_fill_color(self):
        # smoke tests for different color types that should be accepted
        colors = [
            (0.4, 0.2, 0.6),
            [0.4, 0.2, 0.6],
            numpy.array([0.4, 0.2, 0.6]),
            numpy.array([(0.4, 0.2, 0.6)], dtype=rgb_float_dtype)[0],
            (0.4, 0.2, 0.6, 1.0),
            [0.4, 0.2, 0.6, 1.0],
            numpy.array([0.4, 0.2, 0.6, 1.0]),
            numpy.array([(0.4, 0.2, 0.6, 1.0)], dtype=rgba_float_dtype)[0],
        ]
        for color in colors:
            with self.subTest(color=color):
                with self.gc:
                    self.gc.set_fill_color(color)

    # Required methods ####################################################

    @contextlib.contextmanager
    def draw_and_check(self):
        """ A context manager to check the result.
        """
        raise NotImplementedError()

    def create_graphics_context(self, width=300, length=300):
        """ Create the desired graphics context
        """
        raise NotImplementedError()


class DrawingImageTester(DrawingTester):
    """ Basic drawing tests for graphics contexts of gui toolkits.
    """
    def setUp(self):
        self.directory = tempfile.mkdtemp()
        self.filename = os.path.join(self.directory, "rendered")
        self.gc = self.create_graphics_context(600, 600, 2.0)
        self.gc.clear()
        self.gc.set_stroke_color((1.0, 0.0, 0.0))
        self.gc.set_fill_color((1.0, 0.0, 0.0))
        self.gc.set_line_width(5)

    def create_graphics_context(self, width=600, length=600, pixel_scale=1.0):
        """ Create the desired graphics context
        """
        raise NotImplementedError()

    def save_and_return_dpi(self):
        """ Draw an image and save it. Then read it back and return the DPI
        """
        self.gc.begin_path()
        self.gc.arc(150, 150, 100, 0.0, 2 * numpy.pi)
        self.gc.fill_path()

        filename = "{0}.png".format(self.filename)
        self.gc.save(filename)
        with Image.open(filename) as image:
            dpi = image.info['dpi']
        return round(dpi[0])

    @contextlib.contextmanager
    def draw_and_check(self):
        yield
        filename = "{0}.png".format(self.filename)
        self.gc.save(filename)
        self.assertImageSavedWithContent(filename)

    def assertImageSavedWithContent(self, filename):
        """ Load the image and check that there is some content in it.
        """
        image = numpy.array(Image.open(filename))
        # Default is expected to be a totally white image.
        # Therefore we check if the whole image is not white.

        # Previously this method checked for red pixels. However this is
        # not [currently] possible with the quartz backend because it writes
        # out image with premultiplied alpha and none of its pixels are the
        # exact red expected here.

        self.assertEqual(image.shape[:2], (600, 600))
        if image.shape[2] == 3:
            check = numpy.sum(image == [255, 255, 255]) != (600 * 600 * 3)
        elif image.shape[2] == 4:
            check = numpy.sum(image == [255, 255, 255, 255]) != (600 * 600 * 4)
        else:
            self.fail(
                "Pixel size is not 3 or 4, but {0}".format(image.shape[2])
            )
        if check:
            return
        self.fail("The image looks empty, no red pixels were drawn")
