# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
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

from kiva.api import MODERN, Font


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

    def test_text(self):
        with self.draw_and_check():
            font = Font(family=MODERN)
            font.size = 24
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

    # Required methods ####################################################

    @contextlib.contextmanager
    def draw_and_check(self):
        """ A context manager to check the result.
        """
        raise NotImplementedError()

    def create_graphics_context(self, width, length):
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

    def create_graphics_context(self, width, length, pixel_scale):
        """ Create the desired graphics context
        """
        raise NotImplementedError()

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
        # default is expected to be a totally white image

        self.assertEqual(image.shape[:2], (600, 600))
        if image.shape[2] == 3:
            check = numpy.sum(image == [255, 0, 0], axis=2) == 3
        elif image.shape[2] == 4:
            check = numpy.sum(image == [255, 0, 0, 255], axis=2) == 4
        else:
            self.fail(
                "Pixel size is not 3 or 4, but {0}".format(image.shape[2])
            )
        if check.any():
            return
        self.fail("The image looks empty, no red pixels were drawn")
