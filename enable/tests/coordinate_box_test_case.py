import unittest

from enable.api import CoordinateBox

class CoordinateBoxTestCase(unittest.TestCase):
    def check_position(self):
        c = CoordinateBox(bounds=[50.0, 50.0])
        self.assertTrue(c.position[0] == c.x)
        self.assertTrue(c.position[1] == c.y)
        self.assertTrue(c.x == 0.0)
        self.assertTrue(c.y == 0.0)
        return

    def check_bounds(self):
        c = CoordinateBox(bounds=[50.0, 60.0])
        self.assertTrue(c.width == c.bounds[0])
        self.assertTrue(c.height == c.bounds[1])
        self.assertTrue(c.bounds[0] == 50.0)
        self.assertTrue(c.bounds[1] == 60.0)
        self.assertTrue(c.x2 == 49.0)
        self.assertTrue(c.y2 == 59.0)
        return

    def check_is_in(self):
        c = CoordinateBox(x=10, y=20)
        c.width=100
        c.height=100
        self.assertTrue(c.is_in(10, 20))
        self.assertTrue(c.is_in(100, 100))
        self.assertTrue(c.is_in(15, 50))
        self.assertTrue(not c.is_in(0, 0))
        self.assertTrue(not c.is_in(10, 10))
        return

if __name__ == "__main__":
    import nose
    nose.main()

# EOF
