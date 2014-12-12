import contextlib

from kiva.tests.drawing_tester import DrawingTester
from kiva.ps import PSGC
from traits.testing.unittest_tools import unittest


class TestPSDrawing(DrawingTester, unittest.TestCase):

    def create_graphics_context(self, width, height):
        return PSGC((width, height))

    @contextlib.contextmanager
    def draw_and_check(self):
        yield
        filename = "{0}.eps".format(self.filename)
        self.gc.save(filename)
        with open(filename, 'r') as handle:
            lines = handle.readlines()

        # Just a simple check that the path has been closed or the text has
        # been drawn.
        line = lines[-1].strip()
        if not any((
                line.endswith('fill'),
                line.endswith('stroke'),
                line.endswith('cliprestore'),
                '(hello kiva) show\n' in lines)):
            self.fail('Path was not closed')


if __name__ == "__main__":
    unittest.main()
