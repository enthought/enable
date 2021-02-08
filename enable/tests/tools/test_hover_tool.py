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

from pyface.toolkit import toolkit_object

from enable.component import Component
from enable.testing import EnableTestAssistant
from enable.tools.hover_tool import HoverTool

from enable.tests._testing import skip_if_null


GuiTestAssistant = toolkit_object("util.gui_test_assistant:GuiTestAssistant")
no_gui_test_assistant = GuiTestAssistant.__name__ == "Unimplemented"
if no_gui_test_assistant:

    # ensure null toolkit has an inheritable GuiTestAssistant
    # Note that without this definition, the test case fails as soon as it
    # is instantiated, before it can be skipped.
    class GuiTestAssistant(object):
        pass


LOWER_BOUND = 50
SIZE = 100
UPPER_BOUND = LOWER_BOUND + SIZE
MIDDLE = LOWER_BOUND + (SIZE / 2)
LOCATIONS = [
    ("left", LOWER_BOUND + 1, MIDDLE + 1),
    ("top", MIDDLE - 2, UPPER_BOUND - 2),
    ("right", UPPER_BOUND - 2, MIDDLE + 1),
    ("bottom", MIDDLE + 1, LOWER_BOUND + 1),
    ("borders", LOWER_BOUND + 1, LOWER_BOUND + 1),
    ("LL", LOWER_BOUND + 1, LOWER_BOUND + 1),
    ("UL", LOWER_BOUND + 1, UPPER_BOUND - 2),
    ("UR", UPPER_BOUND - 2, UPPER_BOUND - 2),
    ("LR", UPPER_BOUND - 2, LOWER_BOUND + 1),
    ("corners", LOWER_BOUND + 1, LOWER_BOUND + 1),
]


@skip_if_null
@unittest.skipIf(no_gui_test_assistant, "GuiTestAssistant not available.")
class HoverToolTestCase(EnableTestAssistant, GuiTestAssistant,
                        unittest.TestCase):
    def setUp(self):
        super(HoverToolTestCase, self).setUp()
        self.component = Component(
            position=[LOWER_BOUND, LOWER_BOUND], bounds=[SIZE, SIZE]
        )
        # add hover tool with hover zone in lower-left corner of component
        self.tool = HoverTool(component=self.component, area_type="LL")
        self.component.tools.append(self.tool)

    @mock.patch("enable.tools.hover_tool.GetGlobalMousePosition")
    def test_basic_hover(self, mock_mouse_pos):
        for i, (area_type, x_pos, y_pos) in enumerate(LOCATIONS):
            self.component.tools = []
            tool = HoverTool(component=self.component, area_type=area_type)
            self.component.tools.append(tool)
            xy_pos = (x_pos, y_pos)
            mock_mouse_pos.return_value = xy_pos
            with self.assertTraitChanges(tool, "_timer", count=1):
                with self.assertTraitChanges(tool, "_start_xy", count=1):

                    def condition():
                        return mock_mouse_pos.call_count == 2 * (i + 1)

                    with self.event_loop_until_condition(condition):
                        self.mouse_move(self.component, *xy_pos)

            self.assertEqual(mock_mouse_pos.call_count, 2 * (i + 1))
            self.assertEqual(tool.event_state, "normal")

    @mock.patch("enable.tools.hover_tool.GetGlobalMousePosition")
    def test_out_of_hover_zone(self, mock_mouse_pos):
        tool = self.tool
        xy_pos = (145, 145)  # mouse in upper-right corner of component
        mock_mouse_pos.return_value = xy_pos
        with self.assertTraitDoesNotChange(tool, "_timer"):
            with self.assertTraitDoesNotChange(tool, "_start_xy"):
                self.mouse_move(self.component, *xy_pos)

        self.assertEqual(mock_mouse_pos.call_count, 0)
        self.assertEqual(tool.event_state, "normal")
