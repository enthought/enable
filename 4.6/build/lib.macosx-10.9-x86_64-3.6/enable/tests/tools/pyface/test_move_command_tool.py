# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
""" Test case for MoveCommandTool """
import unittest
from unittest.mock import MagicMock

# Enthought library imports
from pyface.undo.api import CommandStack
from traits.testing.api import UnittestTools

# Local library imports
from enable.component import Component
from enable.container import Container
from enable.testing import EnableTestAssistant
from enable.tools.pyface.commands import MoveCommand, ResizeCommand
from enable.tools.pyface.move_command_tool import MoveCommandTool


class MoveCommandToolTestCase(unittest.TestCase, EnableTestAssistant,
                              UnittestTools):
    def setUp(self):
        self.command_stack = CommandStack()
        self.command_stack.push = MagicMock()
        self.component = Component(position=[50, 50], bounds=[100, 100])
        self.container = Container()
        self.container.add(self.component)
        self.tool = MoveCommandTool(
            component=self.component, command_stack=self.command_stack
        )
        self.component.tools.append(self.tool)

    def test_drag_component(self):
        window = self.create_mock_window()

        # start the mouse drag
        mouse_down_event = self.mouse_down(
            self.component, 145, 145, window=window
        )
        self.assertTrue(self.tool._mouse_down_received)

        # start moving the mouse
        mouse_move_event = self.mouse_move(
            self.component, 145, 145, window=window
        )
        self.assertTrue(mouse_move_event.handled)

        # move the mouse to the final location
        mouse_move_event = self.mouse_move(
            self.component, 195, 95, window=window
        )
        self.assertTrue(mouse_move_event.handled)

        # release the mouse, ending the drag
        mouse_up_event = self.mouse_up(self.component, 195, 95, window=window)
        self.assertTrue(mouse_up_event.handled)

        # check if a MoveCommand was pushed onto the stack
        args, kwargs = self.command_stack.push.call_args
        self.assertIsNotNone(args)
        self.assertEqual(len(args), 1)
        self.assertIsInstance(args[0], MoveCommand)
        command = args[0]

        # check that the MoveCommand has right parameters
        self.assertEqual(command.data, (100, 0))
        self.assertEqual(command.previous_position, (50, 50))
        self.assertFalse(command.mergeable)
        self.assertEqual(command.component, self.component)

    def test_drag_cancel(self):
        window = self.create_mock_window()

        # start the mouse drag
        mouse_down_event = self.mouse_down(
            self.component, 145, 145, window=window
        )
        # could be a click, not a drag, I guess?
        self.assertFalse(mouse_down_event.handled)
        self.assertTrue(self.tool._mouse_down_received)

        # start moving the mouse
        mouse_move_event = self.mouse_move(
            self.component, 145, 145, window=window
        )
        self.assertTrue(mouse_move_event.handled)

        # move the mouse to the final location
        mouse_move_event = self.mouse_move(
            self.component, 195, 95, window=window
        )
        self.assertTrue(mouse_move_event.handled)

        # send an escape to cancel the event
        escape_event = self.send_key(self.component, "Esc", window=window)
        self.assertTrue(escape_event.handled)

        # release the mouse, ending the drag
        mouse_up_event = self.mouse_up(self.component, 195, 95, window=window)
        self.assertFalse(mouse_up_event.handled)

        # check if a MoveCommand was pushed onto the stack
        self.assertFalse(self.command_stack.push.called)

    def test_drag_resize_move_command(self):
        self.tool.command = ResizeCommand.move_command

        window = self.create_mock_window()

        # start the mouse drag
        mouse_down_event = self.mouse_down(
            self.component, 145, 145, window=window
        )
        self.assertTrue(self.tool._mouse_down_received)

        # start moving the mouse
        mouse_move_event = self.mouse_move(
            self.component, 145, 145, window=window
        )
        self.assertTrue(mouse_move_event.handled)

        # move the mouse to the final location
        mouse_move_event = self.mouse_move(
            self.component, 195, 95, window=window
        )
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
        self.assertEqual(command.data, (100, 0, 100, 100))
        self.assertEqual(command.previous_rectangle, (50, 50, 100, 100))
        self.assertFalse(command.mergeable)
        self.assertEqual(command.component, self.component)
