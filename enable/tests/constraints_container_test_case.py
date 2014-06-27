import unittest

from enable.api import ConstraintsContainer, Component
from enable.layout.api import hbox, vbox, align, spacer, grid
from enable.layout.layout_helpers import DefaultSpacing


class ConstraintsContainerTestCase(unittest.TestCase):

    def setUp(self):
        self.container = ConstraintsContainer(bounds=[100.0, 100.0])
        self.c1 = Component()
        self.c2 = Component()
        self.container.add(self.c1)
        self.container.add(self.c2)

    def test_equal_size(self):
        """ Test alignment of widths and heights.

        """
        self.container.layout_constraints = [
            self.c1.layout_width == 10,
            self.c2.layout_width == 10,
            self.c1.layout_height == 10,
            self.c2.layout_height == 10
        ]

        self.assert_(self.c1.bounds == self.c2.bounds)

    def test_hbox_order(self):
        """ Test the order of components in an hbox.

        """
        self.container.layout_constraints = [
            hbox(self.c1, self.c2)
        ]

        dx = self.c2.position[0] - self.c1.position[0]
        self.assert_(dx > 0)

    def test_vbox_order(self):
        """ Test the order of components in a vbox.

        """
        self.container.layout_constraints = [
            vbox(self.c1, self.c2)
        ]

        dy = self.c2.position[1] - self.c1.position[1]
        self.assert_(dy < 0)

    def test_alignment_vertical(self):
        """ Test alignment of components vertically with constraints.

        """
        self.container.layout_constraints = [
            self.c1.layout_height == 10,
            self.c2.layout_height == 10,
            align('v_center', self.container, self.c1, self.c2)
        ]

        pos1 = self.c1.position
        bound1 = self.c1.bounds
        pos2 = self.c2.position
        bound2 = self.c2.bounds

        self.assert_(pos1[1] + bound1[1] / 2 == self.container.bounds[1] / 2)
        self.assert_(pos2[1] + bound2[1] / 2 == self.container.bounds[1] / 2)

    def test_alignment_horizontal(self):
        """ Test alignment of components horizontally with constraints.

        """
        self.container.layout_constraints = [
            self.c1.layout_width == 10,
            self.c2.layout_width == 10,
            align('h_center', self.container, self.c1, self.c2)
        ]

        pos1 = self.c1.position
        bound1 = self.c1.bounds
        pos2 = self.c2.position
        bound2 = self.c2.bounds

        self.assert_(pos1[0] + bound1[0] / 2 == self.container.bounds[0] / 2)
        self.assert_(pos2[0] + bound2[0] / 2 == self.container.bounds[0] / 2)

    def test_constraint_function(self):
        """ Test using a function to create constraints.

        """
        cns = [
            hbox(self.c1, self.c2),
            align('layout_width', self.c1, self.c2)
        ]

        def get_constraints(container):
            return cns

        self.container.layout_constraints = get_constraints

        self.assert_(self.c1.bounds[0] == self.c2.bounds[0])

    def test_invalid_layout(self):
        """ Make sure proper exceptions are thrown with an invalid layout.

        """
        self.assertRaises(TypeError, setattr,
                          self.container.layout_constraints,
                          [hbox(self.c1, spacer, spacer)])

    def test_grid_layout(self):
        """ Test the grid layout helper.

        """
        c3 = Component()
        c4 = Component()

        self.container.add(c3)
        self.container.add(c4)

        self.container.layout_constraints = [
            grid([self.c1, self.c2], [c3, c4]),
            align('layout_width', self.c1, self.c2, c3, c4),
            align('layout_height', self.c1, self.c2, c3, c4)
        ]

        space = DefaultSpacing.ABUTMENT
        c2_pos = [self.c1.position[0] + self.c1.bounds[0] + space,
                  self.c1.position[1]]
        self.assert_(self.c2.position == c2_pos)

    def test_invalid_grid_layout(self):
        """ Test an invalid grid layout.

        """
        self.assertRaises(TypeError, setattr,
                          self.container.layout_constraints,
                          [grid([self.c1, spacer])])

    def test_constraint_strength(self):
        """ Test the strength of constraints.

        """
        self.container.layout_constraints = [
            (self.c1.layout_width == 10) | 'weak',
            (self.c1.layout_width == 20) | 'strong'
        ]

        self.assert_(self.c1.bounds[0] == 20)

    def test_share_layout(self):
        """ Test sharing layouts with a child container.

        """
        self.child_container = ConstraintsContainer(bounds=[50, 50])
        c3 = Component()
        self.child_container.add(c3)
        self.container.add(self.child_container)

        self.container.layout_constraints = [
            hbox(self.c1, self.c2, c3),
            align('layout_width', self.c1, self.c2, c3)
        ]

        self.assert_(self.c1.bounds[0] == self.c2.bounds[0] != c3.bounds[0])

        self.child_container.share_layout = True
        self.container.relayout()

        self.assert_(self.c1.bounds[0] == self.c2.bounds[0] == c3.bounds[0])


if __name__ == '__main__':
    unittest.main()
