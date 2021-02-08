# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
import unittest

from enable.api import Component, Container, Viewport


class ViewportTestCase(unittest.TestCase):
    def test_basic_viewport(self):
        container = Container(bounds=[100.0, 100.0])
        component = Component(bounds=[50.0, 50.0], position=[5.0, 5.0])
        container.add(component)
        view = Viewport(
            component=container,
            view_position=[10.0, 10.0],
            view_bounds=[50.0, 50.0],
            position=[0, 0],
            bounds=[50, 50],
        )

        self.assertEqual(view.view_position, [10, 10])
        print(view.components_at(0.0, 0.0), view.view_position)
        self.assertTrue(view.components_at(0.0, 0.0)[0] == component)
        self.assertTrue(view.components_at(44.9, 0.0)[0] == component)
        self.assertTrue(view.components_at(0.0, 44.9)[0] == component)
        self.assertTrue(view.components_at(44.9, 44.9)[0] == component)

        self.assertTrue(view.components_at(46.0, 45.0) == [])
        self.assertTrue(view.components_at(46.0, 0.0) == [])
        self.assertTrue(view.components_at(45.0, 46.0) == [])
        self.assertTrue(view.components_at(0.0, 46.0) == [])

    def test_initial_position(self):
        container = Container(bounds=[100.0, 100.0])
        component = Component(bounds=[50.0, 50.0], position=[5.0, 5.0])
        container.add(component)
        view = Viewport(
            component=container,
            view_bounds=[50.0, 50.0],
            position=[0, 0],
            bounds=[50, 50],
        )
        self.assertEqual(view.view_position, [0, 0])

    def test_initial_position_vanchor_top(self):
        container = Container(bounds=[100.0, 100.0])
        component = Component(bounds=[50.0, 50.0], position=[5.0, 5.0])
        container.add(component)
        view = Viewport(
            component=container,
            vertical_anchor="top",
            view_bounds=[50.0, 50.0],
            position=[0, 0],
            bounds=[50, 50],
        )
        self.assertEqual(view.view_position, [0, 50])

    def test_initial_position_vanchor_center(self):
        container = Container(bounds=[100.0, 100.0])
        component = Component(bounds=[50.0, 50.0], position=[5.0, 5.0])
        container.add(component)
        view = Viewport(
            component=container,
            vertical_anchor="center",
            view_bounds=[50.0, 50.0],
            position=[0, 0],
            bounds=[50, 50],
        )
        self.assertEqual(view.view_position, [0, 25])

    def test_initial_position_hanchor_right(self):
        container = Container(bounds=[100.0, 100.0])
        component = Component(bounds=[50.0, 50.0], position=[5.0, 5.0])
        container.add(component)
        view = Viewport(
            component=container,
            horizontal_anchor="right",
            view_bounds=[50.0, 50.0],
            position=[0, 0],
            bounds=[50, 50],
        )
        self.assertEqual(view.view_position, [50, 0])

    def test_initial_position_hanchor_center(self):
        container = Container(bounds=[100.0, 100.0])
        component = Component(bounds=[50.0, 50.0], position=[5.0, 5.0])
        container.add(component)
        view = Viewport(
            component=container,
            horizontal_anchor="center",
            view_bounds=[50.0, 50.0],
            position=[0, 0],
            bounds=[50, 50],
        )
        self.assertEqual(view.view_position, [25, 0])

    def test_adjust_bounds(self):
        container = Container(bounds=[100.0, 100.0])
        component = Component(bounds=[50.0, 50.0], position=[5.0, 5.0])
        container.add(component)
        view = Viewport(
            component=container,
            view_bounds=[50.0, 50.0],
            view_position=[10, 10],
            position=[0, 0],
            bounds=[50, 50],
        )

        # simple resize
        view.bounds = [60, 60]
        self.assertEqual(view.view_position, [10, 10])

    def test_adjust_bounds_vanchor_top(self):
        container = Container(bounds=[100.0, 100.0])
        component = Component(bounds=[50.0, 50.0], position=[5.0, 5.0])
        container.add(component)
        view = Viewport(
            component=container,
            view_bounds=[50.0, 50.0],
            view_position=[20, 20],
            position=[0, 0],
            bounds=[50, 50],
            vertical_anchor="top",
        )

        # simple resize
        view.bounds = [60, 60]
        self.assertEqual(view.view_position, [20, 10.0])

        # resize beyond bottom
        view.bounds = [80, 80]
        self.assertEqual(view.view_position, [20, -10.0])

        # resize bigger than view
        view.bounds = [120, 120]
        self.assertEqual(view.view_position, [20, -50.0])

    def test_adjust_bounds_vanchor_top_stay_inside(self):
        container = Container(bounds=[100.0, 100.0])
        component = Component(bounds=[50.0, 50.0], position=[5.0, 5.0])
        container.add(component)
        view = Viewport(
            component=container,
            view_bounds=[50.0, 50.0],
            view_position=[20, 20],
            position=[0, 0],
            bounds=[50, 50],
            vertical_anchor="top",
            stay_inside=True,
        )

        # simple resize
        view.bounds = [60, 60]
        self.assertEqual(view.view_position, [20, 10.0])

        # resize beyond bottom
        view.bounds = [80, 80]
        self.assertEqual(view.view_position, [20, 0.0])

        # resize bigger than view
        view.bounds = [120, 120]
        self.assertEqual(view.view_position, [0, -20.0])

    def test_adjust_bounds_vanchor_center(self):
        container = Container(bounds=[100.0, 100.0])
        component = Component(bounds=[50.0, 50.0], position=[5.0, 5.0])
        container.add(component)
        view = Viewport(
            component=container,
            view_bounds=[50.0, 50.0],
            view_position=[20, 20],
            position=[0, 0],
            bounds=[50, 50],
            vertical_anchor="center",
        )

        # simple resize
        view.bounds = [60, 60]
        self.assertEqual(view.view_position, [20, 15.0])

        # resize beyond bottom
        view.bounds = [95, 95]
        self.assertEqual(view.view_position, [20, -2.5])

        # resize bigger than view
        view.bounds = [120, 120]
        self.assertEqual(view.view_position, [20, -15.0])

    def test_adjust_bounds_vanchor_center_stay_inside(self):
        container = Container(bounds=[100.0, 100.0])
        component = Component(bounds=[50.0, 50.0], position=[5.0, 5.0])
        container.add(component)
        view = Viewport(
            component=container,
            view_bounds=[50.0, 50.0],
            view_position=[20, 20],
            position=[0, 0],
            bounds=[50, 50],
            vertical_anchor="center",
            stay_inside=True,
        )

        # simple resize
        view.bounds = [60, 60]
        self.assertEqual(view.view_position, [20, 15.0])

        # resize beyond left
        view.bounds = [95, 95]
        self.assertEqual(view.view_position, [5, 0.0])

        # resize bigger than view
        view.bounds = [120, 120]
        self.assertEqual(view.view_position, [0, -10.0])

    def test_adjust_bounds_hanchor_top(self):
        container = Container(bounds=[100.0, 100.0])
        component = Component(bounds=[50.0, 50.0], position=[5.0, 5.0])
        container.add(component)
        view = Viewport(
            component=container,
            view_bounds=[50.0, 50.0],
            view_position=[20, 20],
            position=[0, 0],
            bounds=[50, 50],
            horizontal_anchor="right",
        )

        # simple resize
        view.bounds = [60, 60]
        self.assertEqual(view.view_position, [10, 20.0])

        # resize beyond left
        view.bounds = [80, 80]
        self.assertEqual(view.view_position, [-10, 20.0])

        # resize bigger than view
        view.bounds = [120, 120]
        self.assertEqual(view.view_position, [-50, 20.0])

    def test_adjust_bounds_hanchor_top_stay_inside(self):
        container = Container(bounds=[100.0, 100.0])
        component = Component(bounds=[50.0, 50.0], position=[5.0, 5.0])
        container.add(component)
        view = Viewport(
            component=container,
            view_bounds=[50.0, 50.0],
            view_position=[20, 20],
            position=[0, 0],
            bounds=[50, 50],
            horizontal_anchor="right",
            stay_inside=True,
        )

        # simple resize
        view.bounds = [60, 60]
        self.assertEqual(view.view_position, [10, 20.0])

        # resize beyond left
        view.bounds = [80, 80]
        self.assertEqual(view.view_position, [0, 20.0])

        # resize bigger than view
        view.bounds = [120, 120]
        self.assertEqual(view.view_position, [-20.0, 0])

    def test_adjust_bounds_hanchor_center(self):
        container = Container(bounds=[100.0, 100.0])
        component = Component(bounds=[50.0, 50.0], position=[5.0, 5.0])
        container.add(component)
        view = Viewport(
            component=container,
            view_bounds=[50.0, 50.0],
            view_position=[20, 20],
            position=[0, 0],
            bounds=[50, 50],
            horizontal_anchor="center",
        )

        # simple resize
        view.bounds = [60, 60]
        self.assertEqual(view.view_position, [15.0, 20])

        # resize beyond left
        view.bounds = [95, 95]
        self.assertEqual(view.view_position, [-2.5, 20])

        # resize bigger than view
        view.bounds = [120, 120]
        self.assertEqual(view.view_position, [-15.0, 20])

    def test_adjust_bounds_hanchor_center_stay_inside(self):
        container = Container(bounds=[100.0, 100.0])
        component = Component(bounds=[50.0, 50.0], position=[5.0, 5.0])
        container.add(component)
        view = Viewport(
            component=container,
            view_bounds=[50.0, 50.0],
            view_position=[20, 20],
            position=[0, 0],
            bounds=[50, 50],
            horizontal_anchor="center",
            stay_inside=True,
        )

        # simple resize
        view.bounds = [60, 60]
        self.assertEqual(view.view_position, [15.0, 20])

        # resize beyond left
        view.height = 95
        view.width = 95
        self.assertEqual(view.view_position, [0.0, 5])

        # resize bigger than view
        view.bounds[0] = 120
        view.bounds[1] = 120
        self.assertEqual(view.view_position, [-10.0, 0])

    def test_adjust_container_bounds_vanchor_top(self):
        container = Container(bounds=[100.0, 100.0])
        component = Component(bounds=[50.0, 50.0], position=[5.0, 5.0])
        container.add(component)
        view = Viewport(
            component=container,
            view_bounds=[50.0, 50.0],
            view_position=[20, 20],
            position=[0, 0],
            bounds=[50, 50],
            vertical_anchor="top",
        )

        # simple resize bigger
        container.bounds = [120, 120]
        self.assertEqual(view.view_position, [20, 40.0])

        # simple resize smaller
        container.height = 90
        container.width = 90
        self.assertEqual(view.view_position, [20, 10.0])

        # simple resize much smaller
        container.bounds[0] = 40
        container.bounds[1] = 40
        self.assertEqual(view.view_position, [20, -40.0])

    def test_adjust_container_bounds_vanchor_top_stay_inside(self):
        container = Container(bounds=[100.0, 100.0])
        component = Component(bounds=[50.0, 50.0], position=[5.0, 5.0])
        container.add(component)
        view = Viewport(
            component=container,
            view_bounds=[50.0, 50.0],
            view_position=[20, 20],
            position=[0, 0],
            bounds=[50, 50],
            vertical_anchor="top",
            stay_inside=True,
        )

        # simple resize bigger
        container.bounds = [120, 120]
        self.assertEqual(view.view_position, [20, 40.0])

        # simple resize smaller
        container.height = 90
        container.width = 90
        self.assertEqual(view.view_position, [20, 10.0])

        # simple resize much smaller
        container.bounds[0] = 40
        container.bounds[1] = 40
        self.assertEqual(view.view_position, [0, -10.0])

    def test_adjust_container_bounds_hanchor_right(self):
        container = Container(bounds=[100.0, 100.0])
        component = Component(bounds=[50.0, 50.0], position=[5.0, 5.0])
        container.add(component)
        view = Viewport(
            component=container,
            view_bounds=[50.0, 50.0],
            view_position=[20, 20],
            position=[0, 0],
            bounds=[50, 50],
            horizontal_anchor="right",
        )

        # simple resize bigger
        container.bounds = [120, 120]
        self.assertEqual(view.view_position, [40, 20.0])

        # simple resize smaller
        container.height = 90
        container.width = 90
        self.assertEqual(view.view_position, [10, 20.0])

        # simple resize much smaller
        container.bounds[0] = 40
        container.bounds[1] = 40
        self.assertEqual(view.view_position, [-40, 20.0])

    def test_adjust_container_bounds_hanchor_right_stay_inside(self):
        container = Container(bounds=[100.0, 100.0])
        component = Component(bounds=[50.0, 50.0], position=[5.0, 5.0])
        container.add(component)
        view = Viewport(
            component=container,
            view_bounds=[50.0, 50.0],
            view_position=[20, 20],
            position=[0, 0],
            bounds=[50, 50],
            horizontal_anchor="right",
            stay_inside=True,
        )

        # simple resize bigger
        container.bounds = [120, 120]
        self.assertEqual(view.view_position, [40, 20])

        # simple resize smaller
        container.height = 90
        container.width = 90
        self.assertEqual(view.view_position, [10, 20])

        # simple resize much smaller
        container.bounds[0] = 40
        container.bounds[1] = 40
        self.assertEqual(view.view_position, [-10, 0])
