import unittest

from enable.api import Component


class ComponentTestCase(unittest.TestCase):
    def test_position(self):
        c = Component(bounds=[50.0, 50.0])
        self.assertTrue(c.position[0] == c.x)
        self.assertTrue(c.position[1] == c.y)
        self.assertTrue(c.x == 0.0)
        self.assertTrue(c.y == 0.0)
        return

    def test_bounds(self):
        c = Component(bounds=[50.0, 60.0])
        self.assertTrue(c.width == c.bounds[0])
        self.assertTrue(c.height == c.bounds[1])
        self.assertTrue(c.bounds[0] == 50.0)
        self.assertTrue(c.bounds[1] == 60.0)
        self.assertTrue(c.x2 == c.x + 50.0 - 1)
        self.assertTrue(c.y2 == c.y + 60.0 - 1)
        return

    def test_get_outer_position(self):
        c = Component(bounds=[50.0, 60.0], padding=10, border_visible=False)
        self.assertTrue(c.outer_x == -10)
        self.assertTrue(c.outer_y == -10)
        self.assertTrue(c.outer_position[0] == -10)
        self.assertTrue(c.outer_position[1] == -10)
        self.assertTrue(c.outer_x2 == 59)
        self.assertTrue(c.outer_y2 == 69)
        self.assertTrue(c.outer_width == 70)
        self.assertTrue(c.outer_height == 80)
        self.assertTrue(c.outer_bounds[0] == 70)
        self.assertTrue(c.outer_bounds[1] == 80)
        return

    def test_set_outer_position(self):
        c = Component(bounds=[50.0, 60.0], padding=10, border_visible=False)
        # Test setting various things
        c.outer_position = [0,0]
        self.assertTrue(c.outer_x == 0)
        self.assertTrue(c.outer_y == 0)
        self.assertTrue(c.x == 10)
        self.assertTrue(c.y == 10)
        self.assertTrue(c.outer_x2 == 69)
        self.assertTrue(c.outer_y2 == 79)
        c.outer_x = 10
        self.assertTrue(c.x == 20)
        self.assertTrue(c.outer_x2 == 79)
        c.outer_x2 = 99
        self.assertTrue(c.outer_x2 == 99)
        self.assertTrue(c.outer_x == 30)
        self.assertTrue(c.x2 == 89)
        self.assertTrue(c.x == 40)
        c.outer_y2 = 99
        self.assertTrue(c.outer_y2 == 99)
        self.assertTrue(c.outer_y == 20)
        self.assertTrue(c.y2 == 89)
        self.assertTrue(c.y == 30)

        return

    def test_border(self):
        c = Component(bounds=[50.0, 60.0],
                      position=[20, 20],
                      padding=10, border_visible=True, border_width=1)
        self.assertTrue(c.outer_x == 10)
        self.assertTrue(c.outer_y == 10)
        self.assertTrue(c.outer_bounds[0] == 70)
        self.assertTrue(c.outer_bounds[1] == 80)
        return

    def check_container(self):
        c = Component()
        self.assertTrue(c.container is None)
        return

if __name__ == "__main__":
    import nose
    nose.main()

# EOF
