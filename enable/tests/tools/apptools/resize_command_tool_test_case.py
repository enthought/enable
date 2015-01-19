#
# (C) Copyright 2015 Enthought, Inc., Austin, TX
# All right reserved.
#
# This file is open source software distributed according to the terms in
# LICENSE.txt
#
""" Test case for ResizeCommandTool """

from __future__ import (division, absolute_import, print_function,
                        unicode_literals)

# Third party library imports
from mock import MagicMock

# Enthought library imports
from apptools.undo.api import CommandStack
from traits.testing.unittest_tools import UnittestTools, unittest

# Local library imports
from enable.component import Component
from enable.container import Container
from enable.testing import EnableTestAssistant
from enable.tools.apptools.commands import ResizeCommand
from enable.tools.apptools.resize_command_tool import ResizeCommandTool


class ResizeCommandToolTestCase(unittest.TestCase, EnableTestAssistant,
                                UnittestTools):

    def setUp(self):
        self.command_stack = CommandStack()
        self.command_stack.push = MagicMock()
        self.component = Component(position=[50, 50], bounds=[100, 100])
        self.container = Container()
        self.container.add(self.component)
        self.tool = ResizeCommandTool(component=self.component,
                                      command_stack=self.command_stack)
        self.component.tools.append(self.tool)

    def test_drag_component(self):
        window = self.create_mock_window()

        # start the mouse drag
        mouse_down_event = self.mouse_down(self.component, 145, 145,
                                           window=window)
        self.assertTrue(mouse_down_event.handled)
        self.assertTrue(self.tool._mouse_down_received)

        # start moving the mouse
        mouse_move_event = self.mouse_move(self.component, 145, 145,
                                           window=window)
        self.assertTrue(mouse_move_event.handled)

        # move the mouse to the final location
        mouse_move_event = self.mouse_move(self.component, 195, 95,
                                           window=window)
        self.assertTrue(mouse_move_event.handled)

        # release the mouse, ending the drag
        mouse_up_event = self.mouse_up(self.component, 195, 95, window=window)
        self.assertTrue(mouse_up_event.handled)

        # check if a ResizeCommand was pushed onto the stack
        args, kwargs = self.command_stack.push.call_args
        self.assertIsNotNone(args)
        self.assertEqual(len(args), 1)
        self.assertIsInstance(args[0], ResizeCommand)
        command = args[0]

        # check that the ResizeCommand has right parameters
        self.assertEqual(command.data, (50, 50, 150, 50))
        self.assertEqual(command.previous_rectangle, (50, 50, 100, 100))
        self.assertFalse(command.mergeable)
        self.assertEqual(command.component, self.component)

    def test_drag_cancel(self):
        window = self.create_mock_window()

        # start the mouse drag
        mouse_down_event = self.mouse_down(self.component, 145, 145,
                                           window=window)
        self.assertTrue(mouse_down_event.handled)
        self.assertTrue(self.tool._mouse_down_received)

        # start moving the mouse
        mouse_move_event = self.mouse_move(self.component, 145, 145,
                                           window=window)
        self.assertTrue(mouse_move_event.handled)

        # move the mouse to the final location
        mouse_move_event = self.mouse_move(self.component, 195, 95,
                                           window=window)
        self.assertTrue(mouse_move_event.handled)

        # send an escape to cancel the event
        escape_event = self.send_key(self.component, 'Esc', window=window)
        self.assertTrue(escape_event.handled)

        # release the mouse, ending the drag
        mouse_up_event = self.mouse_up(self.component, 195, 95, window=window)
        self.assertFalse(mouse_up_event.handled)

        # check if a ResizeCommand was pushed onto the stack
        self.assertFalse(self.command_stack.push.called)
