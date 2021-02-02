import unittest

from kiva.agg import GraphicsContextSystem


class GraphicsContextSystemTestCase(unittest.TestCase):
    def test_creation(self):
        """ Simply create and destroy multiple objects.  This silly
            test crashed when we transitioned from Numeric 23.1 to 23.8.
            That problem is fixed now.
        """
        for i in range(10):
            gc = GraphicsContextSystem((100, 100), "rgba32")
            del gc
