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


LOWER_BOUND = 50
SIZE = 100
UPPER_BOUND = LOWER_BOUND + SIZE
MIDDLE = LOWER_BOUND + (SIZE / 2)
LOCATIONS = [
    ('left', LOWER_BOUND + 1, MIDDLE + 1),
    ('top', MIDDLE - 2, UPPER_BOUND - 2),
    ('right', UPPER_BOUND - 2, MIDDLE + 1),
    ('bottom', MIDDLE + 1, LOWER_BOUND + 1),
    ('borders', LOWER_BOUND + 1, LOWER_BOUND + 1),
    ('LL', LOWER_BOUND + 1, LOWER_BOUND + 1),
    ('UL', LOWER_BOUND + 1, UPPER_BOUND - 2),
    ('UR', UPPER_BOUND - 2, UPPER_BOUND - 2),
    ('LR', UPPER_BOUND - 2, LOWER_BOUND + 1),
    ('corners', LOWER_BOUND + 1, LOWER_BOUND + 1),
]


@skipIf(no_gui_test_assistant, 'No GuiTestAssistant')
class HoverToolTestCase(EnableTestAssistant, GuiTestAssistant, TestCase):

    def setUp(self):
        super(HoverToolTestCase, self).setUp()
        self.component = Component(
            position=[LOWER_BOUND, LOWER_BOUND],
            bounds=[SIZE, SIZE],
        )
        # add hover tool with hover zone in lower-left corner of component
        self.tool = HoverTool(component=self.component, area_type='LL')
        self.component.tools.append(self.tool)

    @patch('enable.tools.hover_tool.GetGlobalMousePosition')
    def test_basic_hover(self, mock_global_mouse_position):
        for i, (area_type, x_pos, y_pos) in enumerate(LOCATIONS):
            self.component.tools = []
            tool = HoverTool(component=self.component, area_type=area_type)
            self.component.tools.append(tool)
            xy_pos = (x_pos, y_pos)
            mock_global_mouse_position.return_value = xy_pos
            with self.assertTraitChanges(tool, '_timer', count=1):
                with self.assertTraitChanges(tool, '_start_xy', count=1):
                    condition = lambda: mock_global_mouse_position.call_count == 2 * (i + 1)
                    with self.event_loop_until_condition(condition):
                        self.mouse_move(self.component, *xy_pos)

            self.assertEqual(
                mock_global_mouse_position.call_count,
                2 * (i + 1),
            )
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
