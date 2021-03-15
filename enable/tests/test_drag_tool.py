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
import warnings

from traits.api import Int

from enable.testing import EnableTestAssistant
from enable.tools.drag_tool import DragTool


class DummyTool(DragTool):

    canceled = Int

    ended = Int

    def drag_cancel(self, event):
        self.canceled += 1
        return True

    def drag_end(self, event):
        self.ended += 1
        return True


class DragToolTestCase(EnableTestAssistant, unittest.TestCase):
    def setUp(self):
        self.tool = DummyTool()

    def test_default_cancel_key(self):
        tool = self.tool
        tool._drag_state = "dragging"  # force dragging state
        event = self.send_key(tool, "Esc")
        self.assertEqual(tool.canceled, 1)
        self.assertTrue(event.handled)

    def test_multiple_cancel_keys(self):
        tool = self.tool

        tool._drag_state = "dragging"  # force dragging state
        tool.cancel_keys = ["a", "Left"]
        event = self.send_key(tool, "Esc")
        self.assertEqual(tool.canceled, 0)
        self.assertFalse(event.handled)

        tool._drag_state = "dragging"  # force dragging state
        event = self.send_key(tool, "a")
        self.assertEqual(tool.canceled, 1)
        self.assertTrue(event.handled)

        tool._drag_state = "dragging"  # force dragging state
        event = self.send_key(tool, "Left")
        self.assertEqual(tool.canceled, 2)
        self.assertTrue(event.handled)

    def test_other_key_pressed(self):
        tool = self.tool
        tool._drag_state = "dragging"  # force dragging state
        event = self.send_key(tool, "Left")
        self.assertEqual(tool.canceled, 0)
        self.assertFalse(event.handled)

    def test_mouse_leave_drag_state(self):

        # When end_drag_on_leave is true then the drag_cancel is called
        # and the _drag_state will be 'nondrag'
        tool = self.tool
        tool.end_drag_on_leave = True
        tool._drag_state = "dragging"  # force dragging state
        with self.assertWarns(DeprecationWarning):
            event = self.mouse_leave(interactor=tool, x=0, y=0)
        self.assertEqual(tool.canceled, 1)
        self.assertEqual(tool._drag_state, "nondrag")
        self.assertTrue(event.handled)

        # When end_drag_on_leave is false then the drag_cancel is not called
        # (i.e. counter is not increased) and the _drag_state will still
        # be 'dragging'
        tool.end_drag_on_leave = False
        tool._drag_state = "dragging"  # force dragging state
        event = self.mouse_leave(interactor=tool, x=0, y=0)
        self.assertEqual(tool.canceled, 1)
        self.assertEqual(tool._drag_state, "dragging")
        self.assertFalse(event.handled)

    def test_on_drag_leave(self):
        # When on_drag_leave is 'cancel' then the drag_cancel is called
        # and the _drag_state will be 'nondrag'
        tool = self.tool
        tool.on_drag_leave = 'cancel'
        tool._drag_state = "dragging"  # force dragging state
        event = self.mouse_leave(interactor=tool, x=0, y=0)
        self.assertEqual(tool.canceled, 1)
        self.assertEqual(tool._drag_state, "nondrag")
        self.assertTrue(event.handled)

        # When on_drag_leave is 'end' then the drag_end is called
        # and the _drag_state will be 'nondrag'
        tool.on_drag_leave = 'end'
        tool._drag_state = "dragging"  # force dragging state
        event = self.mouse_leave(interactor=tool, x=0, y=0)
        self.assertEqual(tool.ended, 1)
        self.assertEqual(tool._drag_state, "nondrag")
        self.assertTrue(event.handled)

    def test_on_drag_leave_no_op(self):
        """ If end_drag_on_leave is set, on_drag_leave trait is ignored. """

        tool = self.tool
        tool.end_drag_on_leave = True
        tool.on_drag_leave = 'end'
        tool._drag_state = "dragging"  # force dragging state
        with self.assertWarns(DeprecationWarning):
            event = self.mouse_leave(interactor=tool, x=0, y=0)

        # end_drag_on_leave should be handled like normal
        self.assertEqual(tool.canceled, 1)
        self.assertEqual(tool._drag_state, "nondrag")
        self.assertTrue(event.handled)

        # even though on_drag_leave = 'end' we do nothing
        self.assertEqual(tool.ended, 0)
