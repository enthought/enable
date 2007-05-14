import unittest
import pdb

from enthought.enable2.api import CoordinateBox

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

def test_suite(level=1):
    suites = []
    if level > 0:
        suites.append(unittest.makeSuite(CoordinateBoxTestCase, 'check_'))
    total_suite = unittest.TestSuite(suites)
    return total_suite

def test(level=10):
    all_tests = test_suite(level)
    runner = unittest.TextTestRunner()
    runner.run(all_tests)
    return runner

if __name__ == "__main__":
    test()

# EOF
