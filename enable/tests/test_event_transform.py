# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!

# Standard library imports
import copy
import unittest

# Enthought library imports
from traits.api import Any, Tuple

# Enable imports
from enable.api import (
    AbstractWindow, BasicEvent, Canvas, Component, Container, Viewport
)


class EnableUnitTest(unittest.TestCase):
    def assert_dims(self, obj, **dims):
        """
        checks that each of the named dimensions of the object are a
        certain value.  e.g.   assert_dims(component, x=5.0, y=7.0).
        """
        for dim, val in dims.items():
            self.assertTrue(getattr(obj, dim) == val)


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


class DummyWindow(AbstractWindow):
    """ A stubbed-out subclass of AbstractWindow that passes on most virtual
    methods instead of raising an exception.
    """

    def _capture_mouse(self):
        pass

    def _release_mouse(self):
        pass

    def _create_mouse_event(self, event):
        return event

    def _redraw(self, *args, **kw):
        pass

    def _get_control_size(self):
        return self._size

    def _create_gc(self, *args, **kw):
        pass

    def _window_paint(self, event):
        pass

    def set_pointer(self, pointer):
        pass

    def _set_timer_interval(self, component, interval):
        pass

    def _set_focus(self):
        pass

    def screen_to_window(self, x, y):
        return (x, y)


class EventTransformTestCase(EnableUnitTest):
    def test_simple_container(self):
        """ Tests event handling of nested containers """
        comp = TestComponent(position=[50, 50])
        inner_container = TestContainer(
            bounds=[100.0, 100.0], position=[50, 50]
        )
        inner_container.add(comp)
        outer_container = TestContainer(bounds=[200, 200])
        outer_container.add(inner_container)

        event = BasicEvent(x=105, y=105)
        outer_container.dispatch(event, "left_down")

        self.assertTrue(comp.last_event.x == 55)
        self.assertTrue(comp.last_event.y == 55)
        self.assertTrue(inner_container.last_event.x == 105)
        self.assertTrue(inner_container.last_event.y == 105)

    def test_viewport_container(self):
        """ Tests event handling of viewports (scaling and translation) """
        comp = TestComponent(position=[20, 20])

        container = TestContainer(bounds=[100, 100], position=[50, 50])
        container.add(comp)

        viewport = Viewport(
            component=container, bounds=[400, 400], position=[30, 30]
        )

        # Test unscaled event
        event = BasicEvent(x=105, y=105)
        viewport.dispatch(event, "left_down")

        self.assertTrue(container.last_event.x == 75)
        self.assertTrue(container.last_event.y == 75)
        self.assertTrue(comp.last_event.x == 25)
        self.assertTrue(comp.last_event.y == 25)

        # Translate the viewport's view_position
        container.last_event = None
        comp.last_event = None
        viewport.view_position = [-10, -10]
        event = BasicEvent(x=115, y=115)
        viewport.dispatch(event, "left_down")

        self.assertTrue(container.last_event.x == 75)
        self.assertTrue(container.last_event.y == 75)
        self.assertTrue(comp.last_event.x == 25)
        self.assertTrue(comp.last_event.y == 25)

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

        self.assertTrue(container.last_event.x == 75)
        self.assertTrue(container.last_event.y == 75)
        self.assertTrue(comp.last_event.x == 25)
        self.assertTrue(comp.last_event.y == 25)

    def test_mouse_capture(self):
        """ Tests saving the event's net_transform as well as the dispatch
        history when capturing a mouse and dispatching subsequent events.
        """

        class MouseCapturingComponent(TestComponent):
            captured_event_pos = Tuple

            def normal_left_down(self, event):
                self.captured_event_pos = (event.x, event.y)
                event.window.set_mouse_owner(self, event.net_transform())

        comp = MouseCapturingComponent(position=[20, 20])

        container = TestContainer(bounds=[100, 100], position=[50, 50])
        container.add(comp)

        viewport = Viewport(
            component=container,
            bounds=[400, 400],
            position=[30, 30],
            fit_window=False,
            resizable="",
        )

        window = DummyWindow(_size=(500, 500))
        window.component = viewport

        # Create the first event (to trigger the mouse capture)
        event = BasicEvent(x=105, y=105, window=window)
        window._handle_mouse_event("left_down", event)

        self.assertTrue(window.mouse_owner == comp)

        # Create the second event
        event = BasicEvent(x=107, y=107, window=window)
        old_pos = comp.captured_event_pos
        window._handle_mouse_event("left_down", event)
        new_pos = comp.captured_event_pos
        self.assertTrue(new_pos[0] == old_pos[0] + 2)
        self.assertTrue(new_pos[1] == old_pos[1] + 2)
