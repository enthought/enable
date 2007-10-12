
# Standard library imports
import copy
import unittest

# Enthought library imports
from enthought.traits.api import Any

# Enable imports
from enthought.enable2.api import BasicEvent, Canvas, Component, Container, \
        Viewport


class EnableUnitTest(unittest.TestCase):

    def assert_dims(self, obj, **dims):
        """
        checks that each of the named dimensions of the object are a
        certain value.  e.g.   assert_dims(component, x=5.0, y=7.0).
        """
        for dim, val in dims.items():
            self.assert_( getattr(obj, dim) == val )
        return

class TestComponent(Component):
    """ A component used for testing event handling.  Most notably, it
    saves a copy of the last event it received.
    """

    # Make some nice default bounds
    bounds = [10, 10]

    last_event = Any

    def _dispatch_stateful_event(self, event, suffix):
        super(TestComponent, self)._dispatch_stateful_event(event, suffix)
        self.last_event = copy.copy(event)

class TestContainer(Container):

    last_event = Any

    def _dispatch_stateful_event(self, event, suffix):
        super(TestContainer, self)._dispatch_stateful_event(event, suffix)
        self.last_event = copy.copy(event)

class TestCanvas(Canvas):

    last_event = Any

    def _dispatch_stateful_event(self, event, suffix):
        super(TestCanvas, self)._dispatch_stateful_event(event, suffix)
        self.last_event = copy.copy(event)



class EventTransformTestCase(EnableUnitTest):
    
    def test_simple_container(self):
        """ Tests event handling of nested containers """
        comp = TestComponent(position=[50,50])
        inner_container = TestContainer(bounds=[100.0, 100.0],
                                        position=[50,50])
        inner_container.add(comp)
        outer_container = TestContainer(bounds=[200,200])
        outer_container.add(inner_container)

        event = BasicEvent(x=105, y=105)
        outer_container.dispatch(event, "left_down")

        self.assert_(comp.last_event.x == 55)
        self.assert_(comp.last_event.y == 55)
        self.assert_(inner_container.last_event.x == 105)
        self.assert_(inner_container.last_event.y == 105)
        return

    def test_viewport_container(self):
        """ Tests event handling of viewports (scaling and translation) """
        comp = TestComponent(position=[20,20])

        container = TestContainer(bounds=[100,100], position=[50,50])
        container.add(comp)

        viewport = Viewport(component=container, bounds=[400,400],
                            position=[30,30])

        # Test unscaled event
        event = BasicEvent(x=105, y=105)
        viewport.dispatch(event, "left_down")

        self.assert_(container.last_event.x == 75)
        self.assert_(container.last_event.y == 75)
        self.assert_(comp.last_event.x == 25)
        self.assert_(comp.last_event.y == 25)

        # Translate the viewport's view_position
        container.last_event = None
        comp.last_event = None
        viewport.view_position = [-10,-10]
        event = BasicEvent(x=115, y=115)
        viewport.dispatch(event, "left_down")

        self.assert_(container.last_event.x == 75)
        self.assert_(container.last_event.y == 75)
        self.assert_(comp.last_event.x == 25)
        self.assert_(comp.last_event.y == 25)

        # Do a zoom
        container.last_event = None
        comp.last_event = None
        # Zoom in by a factor of 2, so view an area that is 200x200.
        viewport.zoom = 2.0
        viewport.enable_zoom = True
        viewport.view_position = [-50, -50]
        viewport.view_bounds = [200, 200]
        event = BasicEvent(x=280, y=280)
        viewport.dispatch(event, "left_down")

        self.assert_(container.last_event.x == 75)
        self.assert_(container.last_event.y == 75)
        self.assert_(comp.last_event.x == 25)
        self.assert_(comp.last_event.y == 25)
        return


def test_suite(level=1):
    suites = []
    if level > 0:
        suites.append(unittest.makeSuite(EventTransformTestCase, 'test_'))
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
