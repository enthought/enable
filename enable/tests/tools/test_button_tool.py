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

from traits.testing.api import UnittestTools

from enable.component import Component
from enable.testing import EnableTestAssistant
from enable.tools.button_tool import ButtonTool


class ButtonToolTestCase(EnableTestAssistant, UnittestTools,
                         unittest.TestCase):
    def setUp(self):
        self.component = Component(position=[50, 50], bounds=[100, 100])
        self.tool = ButtonTool(component=self.component)
        self.component.tools.append(self.tool)

    def test_click_function(self):
        tool = self.tool
        with self.assertTraitDoesNotChange(tool, "down"):
            with self.assertTraitChanges(tool, "clicked", count=1):
                with self.assertTraitDoesNotChange(tool, "checked"):
                    tool.click()

    def test_click_function_togglable(self):
        tool = self.tool
        tool.togglable = True

        with self.assertTraitDoesNotChange(tool, "down"):
            with self.assertTraitChanges(tool, "clicked", count=1):
                with self.assertTraitChanges(tool, "checked", count=1):
                    tool.click()

        self.assertTrue(tool.checked)

    def test_toggle_function_untogglable(self):
        tool = self.tool
        with self.assertTraitDoesNotChange(tool, "down"):
            with self.assertTraitDoesNotChange(tool, "clicked"):
                with self.assertTraitChanges(tool, "checked"):
                    tool.toggle()

        self.assertTrue(tool.checked)

    def test_toggle_function(self):
        tool = self.tool
        tool.togglable = True

        with self.assertTraitDoesNotChange(tool, "down"):
            with self.assertTraitDoesNotChange(tool, "clicked"):
                with self.assertTraitChanges(tool, "checked", count=1):
                    tool.toggle()

        self.assertTrue(tool.checked)

    def test_toggle_function_checked(self):
        tool = self.tool
        tool.togglable = True
        tool.checked = True

        with self.assertTraitDoesNotChange(tool, "down"):
            with self.assertTraitDoesNotChange(tool, "clicked"):
                with self.assertTraitChanges(tool, "checked", count=1):
                    tool.toggle()

        self.assertFalse(tool.checked)

    def test_basic_click(self):
        window = self.create_mock_window()
        component = self.component
        tool = self.tool

        with self.assertTraitChanges(tool, "down", count=1):
            with self.assertTraitDoesNotChange(tool, "clicked"):
                with self.assertTraitDoesNotChange(tool, "checked"):
                    event = self.mouse_down(component, 100, 100, window=window)

        self.assertTrue(event.handled)
        self.assertTrue(tool.event_state, "pressed")
        self.assertTrue(tool.down)
        self.assertFalse(tool.checked)

        with self.assertTraitChanges(tool, "down", count=1):
            with self.assertTraitChanges(tool, "clicked", count=1):
                with self.assertTraitDoesNotChange(tool, "checked"):
                    event = self.mouse_up(
                        self.component, 100, 100, window=window
                    )

        self.assertTrue(event.handled)
        self.assertTrue(tool.event_state, "normal")
        self.assertFalse(tool.down)
        self.assertFalse(tool.checked)

    def test_basic_toggle(self):
        window = self.create_mock_window()
        component = self.component
        tool = self.tool
        tool.togglable = True

        with self.assertTraitChanges(tool, "down", count=1):
            with self.assertTraitDoesNotChange(tool, "clicked"):
                with self.assertTraitDoesNotChange(tool, "checked"):
                    event = self.mouse_down(component, 100, 100, window=window)

        self.assertTrue(event.handled)
        self.assertTrue(tool.event_state, "pressed")
        self.assertTrue(tool.down)
        self.assertFalse(tool.checked)

        with self.assertTraitChanges(tool, "down", count=1):
            with self.assertTraitChanges(tool, "clicked", count=1):
                with self.assertTraitChanges(tool, "checked", count=1):
                    event = self.mouse_up(
                        self.component, 100, 100, window=window
                    )

        self.assertTrue(event.handled)
        self.assertTrue(tool.event_state, "normal")
        self.assertFalse(tool.down)
        self.assertTrue(tool.checked)

    def test_basic_toggle_checked(self):
        window = self.create_mock_window()
        component = self.component
        tool = self.tool
        tool.togglable = True
        tool.checked = True

        with self.assertTraitChanges(tool, "down", count=1):
            with self.assertTraitDoesNotChange(tool, "clicked"):
                with self.assertTraitDoesNotChange(tool, "checked"):
                    event = self.mouse_down(component, 100, 100, window=window)

        self.assertTrue(event.handled)
        self.assertTrue(tool.event_state, "pressed")
        self.assertTrue(tool.down)
        self.assertTrue(tool.checked)

        with self.assertTraitChanges(tool, "down", count=1):
            with self.assertTraitChanges(tool, "clicked", count=1):
                with self.assertTraitChanges(tool, "checked", count=1):
                    event = self.mouse_up(
                        self.component, 100, 100, window=window
                    )

        self.assertTrue(event.handled)
        self.assertTrue(tool.event_state, "normal")
        self.assertFalse(tool.down)
        self.assertFalse(tool.checked)

    def test_basic_click_disabled(self):
        window = self.create_mock_window()
        component = self.component
        tool = self.tool
        tool.enabled = False

        with self.assertTraitDoesNotChange(tool, "down"):
            with self.assertTraitDoesNotChange(tool, "clicked"):
                with self.assertTraitDoesNotChange(tool, "checked"):
                    event = self.mouse_down(component, 100, 100, window=window)

        self.assertFalse(event.handled)
        self.assertTrue(tool.event_state, "normal")
        self.assertFalse(tool.down)
        self.assertFalse(tool.checked)

        with self.assertTraitDoesNotChange(tool, "down"):
            with self.assertTraitDoesNotChange(tool, "clicked"):
                with self.assertTraitDoesNotChange(tool, "checked"):
                    event = self.mouse_up(
                        self.component, 100, 100, window=window
                    )

        self.assertFalse(event.handled)
        self.assertTrue(tool.event_state, "normal")
        self.assertFalse(tool.down)
        self.assertFalse(tool.checked)

    def test_click_drag(self):
        window = self.create_mock_window()
        component = self.component
        tool = self.tool

        with self.assertTraitChanges(tool, "down", count=1):
            with self.assertTraitDoesNotChange(tool, "clicked"):
                with self.assertTraitDoesNotChange(tool, "checked"):
                    event = self.mouse_down(component, 100, 100, window=window)

        self.assertTrue(event.handled)
        self.assertTrue(tool.event_state, "pressed")
        self.assertTrue(tool.down)
        self.assertFalse(tool.checked)

        # move mouse out of component
        with self.assertTraitChanges(tool, "down", count=1):
            with self.assertTraitDoesNotChange(tool, "clicked"):
                with self.assertTraitDoesNotChange(tool, "checked"):
                    event = self.mouse_move(
                        self.component, 25, 25, window=window
                    )

        self.assertFalse(event.handled)
        self.assertTrue(tool.event_state, "pressed")
        self.assertFalse(tool.down)
        self.assertFalse(tool.checked)

        # move mouse into component
        with self.assertTraitChanges(tool, "down", count=1):
            with self.assertTraitDoesNotChange(tool, "clicked"):
                with self.assertTraitDoesNotChange(tool, "checked"):
                    event = self.mouse_move(
                        self.component, 100, 100, window=window
                    )

        self.assertFalse(event.handled)
        self.assertTrue(tool.event_state, "pressed")
        self.assertTrue(tool.down)
        self.assertFalse(tool.checked)

        # move mouse out of component
        with self.assertTraitChanges(tool, "down", count=1):
            with self.assertTraitDoesNotChange(tool, "clicked"):
                with self.assertTraitDoesNotChange(tool, "checked"):
                    event = self.mouse_move(
                        self.component, 200, 200, window=window
                    )

        self.assertFalse(event.handled)
        self.assertTrue(tool.event_state, "pressed")
        self.assertFalse(tool.down)
        self.assertFalse(tool.checked)

        # release button
        with self.assertTraitDoesNotChange(tool, "down"):
            with self.assertTraitDoesNotChange(tool, "clicked"):
                with self.assertTraitDoesNotChange(tool, "checked"):
                    event = self.mouse_up(
                        self.component, 200, 200, window=window
                    )

        self.assertFalse(event.handled)
        self.assertTrue(tool.event_state, "normal")
        self.assertFalse(tool.down)
        self.assertFalse(tool.checked)

    def test_click_mouse_leave(self):
        window = self.create_mock_window()
        component = self.component
        tool = self.tool

        with self.assertTraitChanges(tool, "down", count=1):
            with self.assertTraitDoesNotChange(tool, "clicked"):
                with self.assertTraitDoesNotChange(tool, "checked"):
                    event = self.mouse_down(component, 100, 100, window=window)

        self.assertTrue(event.handled)
        self.assertTrue(tool.event_state, "pressed")
        self.assertTrue(tool.down)
        self.assertFalse(tool.checked)

        # move mouse out of component
        with self.assertTraitChanges(tool, "down", count=1):
            with self.assertTraitDoesNotChange(tool, "clicked"):
                with self.assertTraitDoesNotChange(tool, "checked"):
                    event = self.mouse_leave(
                        self.component, 25, 25, window=window
                    )

        self.assertFalse(event.handled)
        self.assertTrue(tool.event_state, "pressed")
        self.assertFalse(tool.down)
        self.assertFalse(tool.checked)

        # move mouse into component
        with self.assertTraitChanges(tool, "down", count=1):
            with self.assertTraitDoesNotChange(tool, "clicked"):
                with self.assertTraitDoesNotChange(tool, "checked"):
                    event = self.mouse_enter(
                        self.component, 100, 100, window=window
                    )

        self.assertFalse(event.handled)
        self.assertTrue(tool.event_state, "pressed")
        self.assertTrue(tool.down)
        self.assertFalse(tool.checked)
