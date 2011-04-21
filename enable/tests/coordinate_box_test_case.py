import unittest

from enable.api import CoordinateBox

class CoordinateBoxTestCase(unittest.TestCase):
    def check_position(self):
        c = CoordinateBox(bounds=[50.0, 50.0])
        self.assert_(c.position[0] == c.x)
        self.assert_(c.position[1] == c.y)
        self.assert_(c.x == 0.0)
        self.assert_(c.y == 0.0)
        return

    def check_bounds(self):
        c = CoordinateBox(bounds=[50.0, 60.0])
        self.assert_(c.width == c.bounds[0])
        self.assert_(c.height == c.bounds[1])
        self.assert_(c.bounds[0] == 50.0)
        self.assert_(c.bounds[1] == 60.0)
        self.assert_(c.x2 == 49.0)
        self.assert_(c.y2 == 59.0)
        return

    def check_is_in(self):
        c = CoordinateBox(x=10, y=20)
        c.width=100
        c.height=100
        self.assert_(c.is_in(10, 20))
        self.assert_(c.is_in(100, 100))
        self.assert_(c.is_in(15, 50))
        self.assert_(not c.is_in(0, 0))
        self.assert_(not c.is_in(10, 10))
        return

if __name__ == "__main__":
    import nose
    nose.main()

# EOF
