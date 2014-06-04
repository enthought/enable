import unittest

from kiva.testing import KivaTestAssistant


class Drawable(object):

    def __init__(self, should_draw=True, should_process=True):
        self.should_draw = should_draw
        self.should_process = should_process

    def draw(self, gc):
        with gc:
            if self.should_draw:
                gc.move_to(-5,0)
                gc.line_to(5,0)
                gc.move_to(0,5)
                gc.line_to(0,-5)
                gc.move_to(0,0)
                # The path will not be processed and remain in the gc cache
                # if we do not execute the stroke_path command.
                if self.should_process:
                    gc.stroke_path()


class TestKivaTestAssistant(KivaTestAssistant, unittest.TestCase):

    def test_path_created_assertions(self):
        drawable = Drawable(should_draw=False)

        # drawing nothing
        self.assertRaises(AssertionError,
            self.assertPathsAreCreated,
            drawable)

        #drawing something
        drawable.should_draw = True
        self.assertPathsAreCreated(drawable)

    def test_paths_processed_assertions(self):
        drawable = Drawable(should_draw=True, should_process=False)

        # not finishing the path
        self.assertRaises(
            AssertionError,
            self.assertPathsAreProcessed,
            drawable)

        #drawing something
        drawable.should_process = True
        self.assertPathsAreProcessed(drawable)


if __name__ == '__main__':
    unittest.main()
