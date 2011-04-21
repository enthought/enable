import unittest

from enable.api import Component, Container, Viewport


class ViewportTestCase(unittest.TestCase):

    def test_basic_viewport(self):
        container = Container(bounds=[100.0, 100.0])
        component = Component(bounds=[50.0, 50.0], position=[5.0, 5.0])
        container.add(component)
        view = Viewport(component=container,
                        view_position=[10.0, 10.0],
                        view_bounds=[50.0, 50.0],
                        position=[0,0],
                        bounds=[50,50])
        self.assert_(view.components_at(0.0, 0.0)[0] == component)
        self.assert_(view.components_at(44.9, 0.0)[0] == component)
        self.assert_(view.components_at(0.0, 44.9)[0] == component)
        self.assert_(view.components_at(44.9, 44.9)[0] == component)

        self.assert_(view.components_at(46.0, 45.0) == [])
        self.assert_(view.components_at(46.0, 0.0) == [])
        self.assert_(view.components_at(45.0, 46.0) == [])
        self.assert_(view.components_at(0.0, 46.0) == [])
        return


if __name__ == "__main__":
    import nose
    nose.main()

# EOF
