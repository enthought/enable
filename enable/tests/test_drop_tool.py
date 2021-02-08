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

from enable.testing import EnableTestAssistant
from enable.tools.base_drop_tool import BaseDropTool


class DummyTool(BaseDropTool):
    def accept_drop(self, position, obj):
        """ Simple rule which allows testing of different cases

        We accept the drop if x > y

        """
        x, y = position
        return x > y

    def handle_drop(self, position, obj):
        # remember the data for checking later
        self.position = position
        self.obj = obj


class DropToolTestCase(EnableTestAssistant, unittest.TestCase):
    def setUp(self):
        self.tool = DummyTool()

    def test_get_drag_result_accept(self):
        result = self.tool.get_drag_result((50, 100), "object")
        self.assertEqual(result, None)

    def test_get_drag_result_reject(self):
        result = self.tool.get_drag_result((50, 100), "object")
        self.assertEqual(result, None)

    def test_get_drag_result_accept_None(self):
        # if object is None, have to accept to support Wx
        result = self.tool.get_drag_result((50, 100), None)
        self.assertEqual(result, "copy")

    def test_drag_over_accept(self):
        event = self.send_drag_over(self.tool, 100, 50, "object")
        self.assertEqual(event.window._drag_result, "copy")
        self.assertTrue(event.handled)

    def test_drag_over_reject(self):
        event = self.send_drag_over(self.tool, 50, 100, "object")
        self.assertEqual(event.window._drag_result, None)
        self.assertFalse(event.handled)

    def test_drag_over_accept_None(self):
        # if object is None, have to accept to support Wx
        event = self.send_drag_over(self.tool, 50, 100, None)
        self.assertEqual(event.window._drag_result, "copy")
        self.assertTrue(event.handled)

    def test_drag_leave(self):
        # we don't attempt to handle these
        event = self.send_drag_leave(self.tool, 100, 50)
        self.assertFalse(event.handled)

    def test_dropped_on_accept(self):
        event = self.send_dropped_on(self.tool, 100, 50, "object")
        self.assertTrue(event.handled)
        self.assertEqual(self.tool.position, (100, 50))
        self.assertEqual(self.tool.obj, "object")

    def test_dropped_on_rejected(self):
        event = self.send_dropped_on(self.tool, 50, 100, "object")
        self.assertFalse(event.handled)
