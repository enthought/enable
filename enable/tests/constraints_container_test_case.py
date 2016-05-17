import unittest

from enable.component import Component

try:
    import kiwisolver
except ImportError:
    ENABLE_CONSTRAINTS = False
else:
    ENABLE_CONSTRAINTS = True


@unittest.skipIf(not ENABLE_CONSTRAINTS, 'kiwisolver not available')
class ConstraintsContainerTestCase(unittest.TestCase):

    def setUp(self):
        from enable.api import ConstraintsContainer

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
        from enable.layout.api import hbox

        self.container.layout_constraints = [
            hbox(self.c1, self.c2)
        ]

        dx = self.c2.position[0] - self.c1.position[0]
        self.assert_(dx > 0)

    def test_vbox_order(self):
        """ Test the order of components in a vbox.

        """
        from enable.layout.api import vbox

        self.container.layout_constraints = [
            vbox(self.c1, self.c2)
        ]

        dy = self.c2.position[1] - self.c1.position[1]
        self.assert_(dy < 0)

    def test_alignment_vertical(self):
        """ Test alignment of components vertically with constraints.

        """
        from enable.layout.api import align

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
        from enable.layout.api import align

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
        from enable.layout.api import hbox, align

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
        from enable.layout.api import hbox, spacer

        self.assertRaises(TypeError, setattr,
                          self.container.layout_constraints,
                          [hbox(self.c1, spacer, spacer)])

    def test_grid_layout(self):
        """ Test the grid layout helper.

        """
        from enable.layout.api import align, grid
        from enable.layout.layout_helpers import DefaultSpacing

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
        from enable.layout.api import spacer, grid

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
        from enable.api import ConstraintsContainer
        from enable.layout.api import hbox, align

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

    def test_layout_manager_initialize(self):
        """ Ensure that a layout manager can only be initialized once.

        """
        manager = self.container._layout_manager
        self.assertRaises(RuntimeError, manager.initialize, [])

    def test_layout_manager_replace_constraints(self):
        """ Test replacing constraints in the layout manager.

        """
        from enable.layout.api import hbox, vbox
        from enable.layout.layout_manager import LayoutManager

        manager = LayoutManager()
        cns = hbox(self.c1, self.c2).get_constraints(self.container)
        new_cns = vbox(self.c1, self.c2).get_constraints(self.container)

        self.assertRaises(RuntimeError, manager.replace_constraints, cns[0],
                          new_cns[0])

        manager.initialize(cns)
        manager.replace_constraints(cns, new_cns)

        self.assert_(not manager._solver.hasConstraint(cns[0]))
        self.assert_(manager._solver.hasConstraint(new_cns[0]))

    def test_layout_manager_max_size(self):
        """ Test the max_size method of the LayoutManager.

        """
        manager = self.container._layout_manager
        max_size = manager.get_max_size(self.container.layout_width,
                                        self.container.layout_height)
        self.assert_(max_size == (-1, -1))


@unittest.skipIf(not ENABLE_CONSTRAINTS, 'kiwisolver not available')
class GeometryTestCase(unittest.TestCase):

    def test_rect(self):
        """ Test the Rect class.

        """
        from enable.layout.geometry import Rect, Box, Pos, Size

        rect = Rect(10, 20, 60, 40)

        self.assert_(rect.box == Box(20, 70, 60, 10))
        self.assert_(rect.pos == Pos(10, 20))
        self.assert_(rect.size == Size(60, 40))

    def test_rect_f(self):
        """ Test the RectF class.

        """
        from enable.layout.geometry import RectF, BoxF, PosF, SizeF

        rect_f = RectF(10.5, 20.5, 60.5, 40.5)

        self.assert_(rect_f.box == BoxF(20.5, 71.0, 61.0, 10.5))
        self.assert_(rect_f.pos == PosF(10.5, 20.5))
        self.assert_(rect_f.size == SizeF(60.5, 40.5))

    def test_box(self):
        """ Test the Box class.

        """
        from enable.layout.geometry import Rect, Box, Pos, Size

        box = Box(20, 70, 60, 10)
        self.assert_(box == Box((20, 70, 60, 10)))
        self.assert_(box.rect == Rect(10, 20, 60, 40))
        self.assert_(box.pos == Pos(10, 20))
        self.assert_(box.size == Size(60, 40))

    def test_box_f(self):
        """ Test the BoxF class.

        """
        from enable.layout.geometry import RectF, BoxF, PosF, SizeF

        box_f = BoxF(20.5, 71.0, 61.0, 10.5)
        self.assert_(box_f == BoxF((20.5, 71.0, 61.0, 10.5)))
        self.assert_(box_f.rect == RectF(10.5, 20.5, 60.5, 40.5))
        self.assert_(box_f.pos == PosF(10.5, 20.5))
        self.assert_(box_f.size == SizeF(60.5, 40.5))

    def test_pos(self):
        """ Test the Pos class.

        """
        from enable.layout.geometry import Pos

        pos = Pos(10, 20)
        self.assert_(pos.x == 10)
        self.assert_(pos.y == 20)

    def test_pos_f(self):
        """ Test the PosF class.

        """
        from enable.layout.geometry import PosF

        pos_f = PosF(10.5, 20.5)
        self.assert_(pos_f.x == 10.5)
        self.assert_(pos_f.y == 20.5)

    def test_size(self):
        """ Test the Size class.

        """
        from enable.layout.geometry import Size

        size = Size(40, 20)
        self.assert_(size == Size((40, 20)))
        self.assert_(size.width == 40)
        self.assert_(size.height == 20)

    def test_size_f(self):
        """ Test the SizeF class.

        """
        from enable.layout.geometry import SizeF

        size_f = SizeF(40.5, 20.5)
        self.assert_(size_f == SizeF((40.5, 20.5)))
        self.assert_(size_f.width == 40.5)
        self.assert_(size_f.height == 20.5)


if __name__ == '__main__':
    unittest.main()
