import unittest

from enable.api import Component, Container


class EnableUnitTest(unittest.TestCase):

    def assert_dims(self, obj, **dims):
        """
        checks that each of the named dimensions of the object are a
        certain value.  e.g.   assert_dims(component, x=5.0, y=7.0).
        """
        for dim, val in dims.items():
            self.assert_( getattr(obj, dim) == val )
        return


class ContainerTestCase(EnableUnitTest):

    def create_simple_components(self):
        "Returns a container with 3 items in it; used by several tests."
        c1 = Component(bounds=[5.0, 10.0])
        c2 = Component(bounds=[6.0, 10.0])
        c3 = Component(bounds=[7.0, 10.0])
        container = Container(bounds=[100.0, 100.0])
        container.add(c1)
        c1.position = [20, 10]
        container.add(c2)
        c2.position = [40, 10]
        container.add(c3)
        c3.position = [60, 10]
        return container

    def test_add_remove(self):
        container = self.create_simple_components()
        self.assert_(len(container.components) == 3)
        components = container.components
        container.remove(components[0])
        container.remove(components[0])
        container.remove(components[0])
        self.assert_(len(container.components) == 0)
        return

    def test_position(self):
        container = self.create_simple_components()
        components = container.components
        self.assert_(components[0].position == [20,10])
        self.assert_(components[1].position == [40,10])
        self.assert_(components[2].position == [60,10])
        return

    def test_position_bounds(self):
        container = Container(bounds=[100.0, 100.0])
        self.assert_dims(container, x=0.0, y=0.0, width=100.0, height=100.0)
        return

    def test_auto_size(self):
        container = Container(bounds=[100.0, 100.0])
        self.assert_(container.auto_size == False)

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
        return


if __name__ == "__main__":
    import nose
    nose.main()

# EOF
