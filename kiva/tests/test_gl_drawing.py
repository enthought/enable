import contextlib

try:
    import pyglet
except ImportError:
    PYGLET_NOT_AVAILABLE = True
else:
    PYGLET_NOT_AVAILABLE = False


from kiva.tests.drawing_tester import DrawingImageTester
from traits.testing.unittest_tools import unittest


@unittest.skipIf(PYGLET_NOT_AVAILABLE, "Cannot import pyglet")
class TestGLDrawing(DrawingImageTester, unittest.TestCase):

    def tearDown(self):
        if hasattr(self, 'window') and self.window is not None:
            self.window.close()
            del self.window
        DrawingImageTester.tearDown(self)

    def create_graphics_context(self, width, height):
        from kiva.gl import GraphicsContext
        self.window = pyglet.window.Window(width=width, height=height)
        gc = GraphicsContext((width, height))
        gc.gl_init()
        return gc

    @unittest.skip("gl graphics context does not support star_clip (#164)")
    def test_star_clip(self):
        # FIXME: overriding test since it segfaults
        DrawingImageTester.test_star_clip(self)

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
        buffer.get_color_buffer().save(filename, encoder=PNGImageEncoder())
        self.assertImageSavedWithContent(filename)


if __name__ == "__main__":
    unittest.main()
