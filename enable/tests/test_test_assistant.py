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
from unittest import mock

from enable.component import Component
from enable.testing import EnableTestAssistant, _MockWindow
from enable.tests._testing import skip_if_null


class TestTestAssistant(unittest.TestCase):
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

    def test_mouse_down(self):
        test_assistant = EnableTestAssistant()
        component = Component(bounds=[100, 200])
        component.normal_left_down = mock.Mock()
        test_assistant.mouse_down(component, x=0, y=0)
        component.normal_left_down.assert_called_once()

    def test_mouse_dclick(self):
        test_assistant = EnableTestAssistant()
        component = Component(bounds=[100, 200])
        component.normal_left_dclick = mock.Mock()
        test_assistant.mouse_dclick(component, x=0, y=0)
        component.normal_left_dclick.assert_called_once()

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

        patch = mock.patch.object(
            Window, "get_pointer_position", return_value=None
        )
        with patch:
            window = Window(None, component=component)
            event = test_assistant.mouse_move(component, 10, 20, window)

            self.assertEqual(event.x, 10)
            self.assertEqual(event.y, 20)
            self.assertEqual(event.window, window)
            self.assertFalse(event.alt_down)
            self.assertFalse(event.control_down)
            self.assertFalse(event.shift_down)
            self.assertEqual(event.window.get_pointer_position(), (10, 20))
