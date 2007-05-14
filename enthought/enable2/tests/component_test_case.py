import unittest

from enthought.enable2.api import Component


class ComponentTestCase(unittest.TestCase):
    def test_position(self):
        c = Component(bounds=[50.0, 50.0])
        self.assert_(c.position[0] == c.x)
        self.assert_(c.position[1] == c.y)
        self.assert_(c.x == 0.0)
        self.assert_(c.y == 0.0)
        return
    
    def test_bounds(self):
        c = Component(bounds=[50.0, 60.0])
        self.assert_(c.width == c.bounds[0])
        self.assert_(c.height == c.bounds[1])
        self.assert_(c.bounds[0] == 50.0)
        self.assert_(c.bounds[1] == 60.0)
        self.assert_(c.x2 == c.x + 50.0 - 1)
        self.assert_(c.y2 == c.y + 60.0 - 1)
        return

    def test_get_outer_position(self):
        c = Component(bounds=[50.0, 60.0], padding=10, border_visible=False)
        self.assert_(c.outer_x == -10)
        self.assert_(c.outer_y == -10)
        self.assert_(c.outer_position[0] == -10)
        self.assert_(c.outer_position[1] == -10)
        self.assert_(c.outer_x2 == 59)
        self.assert_(c.outer_y2 == 69)
        self.assert_(c.outer_width == 70)
        self.assert_(c.outer_height == 80)
        self.assert_(c.outer_bounds[0] == 70)
        self.assert_(c.outer_bounds[1] == 80)
        return
    
    def test_set_outer_position(self):
        c = Component(bounds=[50.0, 60.0], padding=10, border_visible=False)
        # Test setting various things
        c.outer_position = [0,0]
        self.assert_(c.outer_x == 0)
        self.assert_(c.outer_y == 0)
        self.assert_(c.x == 10)
        self.assert_(c.y == 10)
        self.assert_(c.outer_x2 == 69)
        self.assert_(c.outer_y2 == 79)
        c.outer_x = 10
        self.assert_(c.x == 20)
        self.assert_(c.outer_x2 == 79)
        return

    def test_border(self):
        c = Component(bounds=[50.0, 60.0], 
                      position=[20, 20],
                      padding=10, border_visible=True, border_width=1)
        self.assert_(c.outer_x == 9)
        self.assert_(c.outer_y == 9)
        self.assert_(c.outer_bounds[0] == 72)
        self.assert_(c.outer_bounds[1] == 82)
        return

    def check_container(self):
        c = Component()
        self.assert_(c.container is None)
        return

def test_suite(level=1):
    suites = []
    if level > 0:
        suites.append(unittest.makeSuite(ComponentTestCase, 'test_'))
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
