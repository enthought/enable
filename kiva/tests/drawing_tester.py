import contextlib
import os
import shutil
import tempfile

import numpy
from PIL import Image

from kiva.fonttools import Font
from kiva.constants import MODERN

class DrawingTester(object):

    def setUp(self):
        self.directory = tempfile.mkdtemp()
        self.filename = os.path.join(self.directory, 'rendered_image.png')
        self.gc = self.create_graphics_context(300, 300)
        self.gc.clear()

    def tearDown(self):
        shutil.rmtree(self.directory)

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
            self.gc.rect(0,0,200,200)
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

    def assertImageSavedWithContent(self, filename):
        """ Load the image and check that there is some content in it.

        """
        image = numpy.array(Image.open(filename))
        # default is expected to be a totally white image
        mask = image != 255
        if not mask.any():
            self.fail('An empty image was saved')

    @contextlib.contextmanager
    def draw_and_check(self):
        yield
        self.save_to_file()
        self.assertImageSavedWithContent(self.filename)

    #### Required methods ####################################################

    def create_graphics_context(self, width, length):
        """ Create the desired graphics context

        """
        raise NotImplementedError()

    def save_to_file(self):
        """ Save the rendered image into a file.

        The filename is given by self.filename.

        """
        raise NotImplementedError()

