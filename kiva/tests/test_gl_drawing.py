import contextlib

import pyglet

from kiva.tests.drawing_tester import DrawingImageTester
from kiva.gl import GraphicsContext
from traits.testing.unittest_tools import unittest


class TestGLDrawing(DrawingImageTester, unittest.TestCase):

    def tearDown(self):
        if hasattr(self, 'window') and self.window is not None:
            self.window.close()
            del self.window
        DrawingImageTester.tearDown(self)

    def create_graphics_context(self, width, height):
        self.window = pyglet.window.Window(width=width, height=height)
        gc = GraphicsContext((width, height))
        gc.gl_init()
        return gc

    @unittest.skip("gl graphics context does not support start_clip")
    def test_star_clip(self):
        # FIXME: overriding test since it segfaults
        pass

    @contextlib.contextmanager
    def draw_and_check(self):
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        yield
        self.window.dispatch_events()
        filename = "{0}.png".format(self.filename)
        print filename
        pyglet.image.get_buffer_manager().get_color_buffer().save('filename.png')
        self.assertImageSavedWithContent('filename.png')


if __name__ == "__main__":
    unittest.main()
