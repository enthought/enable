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
import sys
import unittest

try:
    import pyglet
except ImportError:
    PYGLET_NOT_AVAILABLE = True
else:
    PYGLET_NOT_AVAILABLE = False

from kiva.tests.drawing_tester import DrawingImageTester

is_windows = sys.platform in ("win32", "cygwin")


@unittest.skipIf(is_windows, "Pyglet/GL backend issues on Windows")
@unittest.skipIf(PYGLET_NOT_AVAILABLE, "Cannot import pyglet")
class TestGLDrawing(DrawingImageTester, unittest.TestCase):
    def tearDown(self):
        if hasattr(self, "window") and self.window is not None:
            self.window.close()
            del self.window
        DrawingImageTester.tearDown(self)

    def create_graphics_context(self, width, height, pixel_scale):
        from kiva.gl import GraphicsContext

        # XXX: Ignore scaling in the unit tests so this works on CI.
        # But really, we should just get rid of this rotted backend.
        self.window = pyglet.window.Window(width=width, height=height)
        gc = GraphicsContext((width, height), base_pixel_scale=1.0)
        gc.gl_init()
        return gc

    @unittest.expectedFailure
    def test_image(self):
        """ gl image drawing is broken. It depends on pygarrayimage, which
        is not actively maintained and appears to be broken now.
        """
        DrawingImageTester.test_image()

    @unittest.skip("gl graphics context does not support star_clip (#164)")
    def test_star_clip(self):
        # FIXME: overriding test since it segfaults
        DrawingImageTester.test_star_clip(self)

    @unittest.skipIf(
        sys.platform == "darwin",
        "Error getting sfnt font name on OSX (enthought/enable#541)",
    )
    def test_text(self):
        DrawingImageTester.test_text(self)

    @unittest.skip("gl graphics context does not clip text properly (#165)")
    def test_text_clip(self):
        # gl graphics context does not clip text properly (#165).
        # may actually work under Wx?
        DrawingImageTester.test_text_clip(self)

    @contextlib.contextmanager
    def draw_and_check(self):
        from pyglet.image.codecs.png import PNGImageEncoder

        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        yield
        self.window.dispatch_events()
        filename = "{0}.png".format(self.filename)
        buffer = pyglet.image.get_buffer_manager()
        with open(filename, mode="wb") as file_handle:
            buffer.get_color_buffer().save(
                filename, file=file_handle, encoder=PNGImageEncoder()
            )
        self.assertImageSavedWithContent(filename)
