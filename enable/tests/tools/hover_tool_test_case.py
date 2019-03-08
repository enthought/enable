# (C) Copyright 2008-2019 Enthought, Inc., Austin, TX
# All rights reserved.

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import time
from unittest import TestCase, skipIf
from unittest.mock import patch

from pyface.toolkit import toolkit_object

from enable.component import Component
from enable.testing import EnableTestAssistant
from enable.tools.hover_tool import HoverTool


GuiTestAssistant = toolkit_object('util.gui_test_assistant:GuiTestAssistant')
no_gui_test_assistant = (GuiTestAssistant.__name__ == 'Unimplemented')


@skipIf(no_gui_test_assistant, 'No GuiTestAssistant')
class HoverToolTestCase(EnableTestAssistant, GuiTestAssistant, TestCase):

    def setUp(self):
        super(HoverToolTestCase, self).setUp()
        self.component = Component(position=[50, 50], bounds=[100, 100])
        # add hover tool with hover zone in lower-left corner of component
        self.tool = HoverTool(component=self.component, area_type='LL')
        self.component.tools.append(self.tool)

    @patch('enable.tools.hover_tool.GetGlobalMousePosition')
    def test_basic_hover(self, mock_global_mouse_position):
        tool = self.tool
        xy_pos = (51, 51)  # mouse in lower-left corner of component
        mock_global_mouse_position.return_value = xy_pos
        with self.assertTraitChanges(tool, '_timer', count=1):
            with self.assertTraitChanges(tool, '_start_xy', count=1):
                condition = lambda: mock_global_mouse_position.call_count == 2
                with self.event_loop_until_condition(condition):
                    self.mouse_move(self.component, *xy_pos)

        self.assertEqual(mock_global_mouse_position.call_count, 2)
        self.assertEqual(tool.event_state, 'normal')

    @patch('enable.tools.hover_tool.GetGlobalMousePosition')
    def test_out_of_hover_zone(self, mock_global_mouse_position):
        tool = self.tool
        xy_pos = (145, 145)  # mouse in upper-right corner of component
        mock_global_mouse_position.return_value = xy_pos
        with self.assertTraitDoesNotChange(tool, '_timer'):
            with self.assertTraitDoesNotChange(tool, '_start_xy'):
                    self.mouse_move(self.component, *xy_pos)

        self.assertEqual(mock_global_mouse_position.call_count, 0)
        self.assertEqual(tool.event_state, 'normal')
