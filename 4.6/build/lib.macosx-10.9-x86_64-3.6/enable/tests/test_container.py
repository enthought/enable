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

from enable.api import Component, Container


class EnableUnitTest(unittest.TestCase):
    def assert_dims(self, obj, **dims):
        """
        checks that each of the named dimensions of the object are a
        certain value.  e.g.   assert_dims(component, x=5.0, y=7.0).
        """
        for dim, val in dims.items():
            self.assertTrue(getattr(obj, dim) == val)


class ContainerTestCase(EnableUnitTest):
    def create_simple_components(self):
        "Returns a container with 3 items in it; used by several tests."
        self.c1 = Component(bounds=[5.0, 10.0])
        self.c2 = Component(bounds=[6.0, 10.0])
        self.c3 = Component(bounds=[7.0, 10.0])
        container = Container(bounds=[100.0, 100.0])
        container.add(self.c1)
        self.c1.position = [20, 10]
        container.add(self.c2)
        self.c2.position = [40, 10]
        container.add(self.c3)
        self.c3.position = [60, 10]
        return container

    def test_get_set_components(self):
        container = self.create_simple_components()
        # Exercise get_components:
        self.assertEqual(container.components, [self.c1, self.c2, self.c3])

        # Exercise set_components:
        new_list = [self.c1, self.c3]
        container.components = new_list
        self.assertEqual(container.components, new_list)

    def test_add_remove(self):
        container = self.create_simple_components()
        self.assertTrue(len(container.components) == 3)
        components = container.components
        container.remove(components[0])
        container.remove(components[0])
        container.remove(components[0])
        self.assertTrue(len(container.components) == 0)

    def test_position(self):
        container = self.create_simple_components()
        components = container.components
        self.assertTrue(components[0].position == [20, 10])
        self.assertTrue(components[1].position == [40, 10])
        self.assertTrue(components[2].position == [60, 10])

    def test_position_bounds(self):
        container = Container(bounds=[100.0, 100.0])
        self.assert_dims(container, x=0.0, y=0.0, width=100.0, height=100.0)

    def test_auto_size(self):
        container = Container(bounds=[100.0, 100.0])
        self.assertFalse(container.auto_size)

        # Add some components
        c1 = Component(position=[10.0, 10.0], bounds=[50.0, 60.0])
        c2 = Component(position=[15.0, 15.0], bounds=[10.0, 10.0])
        container.add(c1)
        container.add(c2)
        self.assert_dims(container, x=0.0, y=0.0, width=100.0, height=100.0)

        # Turn on auto-sizing
        container.auto_size = True
        self.assert_dims(container, x=10.0, y=10.0, width=49.0, height=59.0)

        # Check that the components' positions changed appropriately
        self.assert_dims(c1, x=0.0, y=0.0)
        self.assert_dims(c2, x=5.0, y=5.0)

        # Move the second component
        c2.position = [100.0, 100.0]
        self.assert_dims(container, x=10.0, y=10.0, width=109.0, height=109.0)
        self.assert_dims(c2, x=100.0, y=100.0)

        # Delete the second component
        container.remove(c2)
        self.assert_dims(container, x=10.0, y=10.0, width=49.0, height=59.0)
