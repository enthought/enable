import contextlib
from xml.etree import ElementTree

from kiva.tests.drawing_tester import DrawingTester
from kiva.svg import GraphicsContext
from traits.testing.unittest_tools import unittest


class TestSVGDrawing(DrawingTester, unittest.TestCase):

    def create_graphics_context(self, width, height):
        return GraphicsContext((width, height))

    @contextlib.contextmanager
    def draw_and_check(self):
        yield
        filename = "{0}.svg".format(self.filename)
        self.gc.save(filename)
        tree = ElementTree.parse(filename)
        elements = [element for element in tree.getiterator()]
        if not len(elements) in [4, 7]:
            self.fail('The expected number of elements was not found')


if __name__ == "__main__":
    unittest.main()
