=========================
Testing Enable Components
=========================

In order to assist in the testing of enable :class:`~.Component` and
:class:`~.Interactor` objects, Enable provides the
:class:`~.EnableTestAssistant` class. This is a mixin class intended to work
along side :class:`python:unittest.TestCase`. Often times in order to
effectively test an enable component and its supported interactivity, one needs
to simulate user interactions such as moving/clicking/dragging the mouse, using
keys, etc. This involves manually creating the corresponding events with
appropriate state and dispatching them appropriately. This can be rather
tedious and :class:`~.EnableTestAssistant` provides a number of helper methods
to greatly simplify the process, making tests both faster/cleaner to write and
easier to read. Furthermore, in test scenarios it is ofen unnecessary to launch
the full application window and as such :class:`~.EnableTestAssistant` provides
a :meth:`~.create_mock_window` method for simply
mocking out the window itself. This allows for specifically testing the
component of interest alone, as is the goal in a unit test. Please see the api
docs (:class:`~.EnableTestAssistant`) for the full list of methods
available.

Here is an example

::

    import unittest
    from unittest import mock

    from enable.api import Component
    from enable.testing import EnableTestAssistant

    class TestExample(unittest.TestCase):
        def test_example(self):
            test_assistant = EnableTestAssistant()
            component = Component(bounds=[100, 200])

            event = test_assistant.mouse_move(component, 10, 20)
            self.assertEqual(event.x, 10)
            self.assertEqual(event.y, 20)
            self.assertFalse(event.alt_down)
            self.assertFalse(event.control_down)
            self.assertFalse(event.shift_down)
            self.assertFalse(sevent.left_down)
            self.assertEqual(event.window.get_pointer_position(), (10, 20))

            component.normal_left_down = mock.Mock()
            test_assistant.mouse_down(component, x=0, y=0)
            component.normal_left_down.assert_called_once()

            event = test_assistant.mouse_move(component, 20, 30, left_down=True)
            self.assertEqual(event.x, 20)
            self.assertEqual(event.y, 30)
            self.assertIs(event.left_down, True)
