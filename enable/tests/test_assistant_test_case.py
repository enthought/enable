import mock
import unittest

from traitsui.tests._tools import skip_if_null

from enable.component import Component
from enable.testing import EnableTestAssistant, _MockWindow


class TestAssistantTestCase(unittest.TestCase):

    def test_mouse_move(self):
        test_assistant = EnableTestAssistant()
        component = Component(bounds=[100, 200])
        event = test_assistant.mouse_move(component, 10, 20)

        self.assertEqual(event.x, 10)
        self.assertEqual(event.y, 20)
        self.assertIsInstance(event.window, _MockWindow)
        self.assertFalse(event.alt_down)
        self.assertFalse(event.control_down)
        self.assertFalse(event.shift_down)
        self.assertEqual(event.window.get_pointer_position(), (10, 20))

    @skip_if_null
    def test_mouse_move_real_window(self):
        from enable.api import Window

        test_assistant = EnableTestAssistant()
        component = Component(bounds=[100, 200])
        window = Window(None, component=component)

        event = test_assistant.mouse_move(component, 10, 20, window)

        self.assertEqual(event.x, 10)
        self.assertEqual(event.y, 20)
        self.assertEqual(event.window, window)
        self.assertFalse(event.alt_down)
        self.assertFalse(event.control_down)
        self.assertFalse(event.shift_down)
        # can't test pointer position, not set, but if we get here it didn't
        # try to set the pointer position

    @skip_if_null
    def test_mouse_move_real_window_mocked_position(self):
        from enable.api import Window

        test_assistant = EnableTestAssistant()
        component = Component(bounds=[100, 200])

        with mock.patch.object(Window, 'get_pointer_position',
                               return_value=None):
            window = Window(None, component=component)
            event = test_assistant.mouse_move(component, 10, 20, window)

            self.assertEqual(event.x, 10)
            self.assertEqual(event.y, 20)
            self.assertEqual(event.window, window)
            self.assertFalse(event.alt_down)
            self.assertFalse(event.control_down)
            self.assertFalse(event.shift_down)
            self.assertEqual(event.window.get_pointer_position(), (10, 20))
