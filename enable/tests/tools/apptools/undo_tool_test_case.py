#
# (C) Copyright 2015 Enthought, Inc., Austin, TX
# All right reserved.
#
# This file is open source software distributed according to the terms in
# LICENSE.txt
#
""" Tests for Commands that work with Components """

from __future__ import (division, absolute_import, print_function,
                        unicode_literals)

# Third party library imports
from mock import MagicMock

# Enthought library imports
from apptools.undo.api import UndoManager
from traits.testing.unittest_tools import UnittestTools, unittest

# Local library imports
from enable.base_tool import KeySpec
from enable.component import Component
from enable.testing import EnableTestAssistant
from enable.tools.apptools.undo_tool import UndoTool


class UndoToolTestCase(unittest.TestCase, EnableTestAssistant, UnittestTools):

    def setUp(self):
        self.undo_manager = UndoManager()
        self.undo_manager.undo = MagicMock()
        self.undo_manager.redo = MagicMock()
        self.component = Component()
        self.tool = UndoTool(component=self.component,
                             undo_manager=self.undo_manager)
        self.component.tools.append(self.tool)

    def test_undo_key_press(self):
        key_event = self.create_key_press('z', control_down=True)
        self.component.dispatch(key_event, 'key_pressed')

        self.assertTrue(self.undo_manager.undo.called)
        self.assertFalse(self.undo_manager.redo.called)
        self.assertTrue(key_event.handled)

    def test_redo_key_press(self):
        key_event = self.create_key_press('z', control_down=True,
                                          shift_down=True)
        self.component.dispatch(key_event, 'key_pressed')

        self.assertTrue(key_event.handled)
        self.assertFalse(self.undo_manager.undo.called)
        self.assertTrue(self.undo_manager.redo.called)

    def test_other_key_press(self):
        key_event = self.create_key_press('z')
        self.component.dispatch(key_event, 'key_pressed')

        self.assertFalse(self.undo_manager.undo.called)
        self.assertFalse(self.undo_manager.redo.called)
        self.assertFalse(key_event.handled)

    def test_undo_key_press_non_default(self):
        self.tool.undo_keys = [KeySpec('Left', 'control')]
        key_event = self.create_key_press('Left', control_down=True)
        self.component.dispatch(key_event, 'key_pressed')

        self.assertTrue(self.undo_manager.undo.called)
        self.assertFalse(self.undo_manager.redo.called)
        self.assertTrue(key_event.handled)

    def test_redo_key_press_non_default(self):
        self.tool.redo_keys = [KeySpec('Right', 'control')]
        key_event = self.create_key_press('Right', control_down=True)
        self.component.dispatch(key_event, 'key_pressed')

        self.assertTrue(key_event.handled)
        self.assertFalse(self.undo_manager.undo.called)
        self.assertTrue(self.undo_manager.redo.called)
