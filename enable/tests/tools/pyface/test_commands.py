# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
""" Tests for Commands that work with Components """
import unittest
from unittest.mock import MagicMock

# Enthought library imports
from traits.testing.api import UnittestTools

# Local library imports
from enable.component import Component
from enable.testing import EnableTestAssistant
from enable.tools.pyface.commands import (
    ComponentCommand,
    MoveCommand,
    ResizeCommand,
)


class ComponentCommandTest(unittest.TestCase, EnableTestAssistant,
                           UnittestTools):
    def setUp(self):
        self.component = Component()
        self.command = ComponentCommand(component=self.component)

    def test_name_default(self):
        self.assertEqual(self.command.component_name, "Component")

    def test_name_empty(self):
        self.command.component = None
        self.assertEqual(self.command.component_name, "")


class ResizeCommandTest(unittest.TestCase, EnableTestAssistant, UnittestTools):
    def setUp(self):
        self.component = Component(position=[50, 50], bounds=[100, 100])
        self.component.request_redraw = MagicMock()
        self.command = ResizeCommand(
            self.component, (25, 25, 150, 150), (50, 50, 100, 100)
        )

    def test_name_default(self):
        self.assertEqual(self.command.name, "Resize Component")

    def test_name_alternate_component_name(self):
        self.command.component_name = "My Component"
        self.assertEqual(self.command.name, "Resize My Component")

    def test_do(self):
        command = self.command

        changes = [[self.component], ["position", "bounds"], []]
        with self.assertMultiTraitChanges(*changes):
            command.do()

        self.assertEqual(self.component.position, [25, 25])
        self.assertEqual(self.component.bounds, [150, 150])
        self.assertTrue(self.component.request_redraw.called)

    def test_do_no_previous_rectangle(self):
        command = ResizeCommand(self.component, (25, 25, 150, 150))
        self.assertEqual(command.previous_rectangle, (50, 50, 100, 100))

    def test_do_no_new_position(self):
        with self.assertRaises(TypeError):
            ResizeCommand(
                self.component, previous_rectangle=(50, 50, 100, 100)
            )

    def test_do_no_new_position_with_data(self):
        command = ResizeCommand(
            self.component,
            previous_rectangle=(50, 50, 100, 100),
            data=(25, 25, 150, 150),
        )
        self.assertEqual(command.data, (25, 25, 150, 150))

    def test_do_no_new_position_no_previous_position_with_data(self):
        command = ResizeCommand(self.component, data=(25, 25, 150, 150))
        self.assertEqual(command.data, (25, 25, 150, 150))
        self.assertEqual(command.previous_rectangle, (50, 50, 100, 100))

    def test_move_command(self):
        command = ResizeCommand.move_command(
            self.component, (25, 25), (50, 50)
        )
        self.assertEqual(command.data, (25, 25, 100, 100))
        self.assertEqual(command.previous_rectangle, (50, 50, 100, 100))

    def test_move_command_no_previous_position(self):
        command = ResizeCommand.move_command(self.component, (25, 25))
        self.assertEqual(command.data, (25, 25, 100, 100))
        self.assertEqual(command.previous_rectangle, (50, 50, 100, 100))

    def test_undo(self):
        command = self.command
        command.do()

        changes = [[self.component], ["position", "bounds"], []]
        with self.assertMultiTraitChanges(*changes):
            command.undo()

        self.assertEqual(self.component.position, [50, 50])
        self.assertEqual(self.component.bounds, [100, 100])
        self.assertTrue(self.component.request_redraw.called)

    def test_redo(self):
        command = self.command
        command.do()
        command.undo()

        changes = [[self.component], ["position", "bounds"], []]
        with self.assertMultiTraitChanges(*changes):
            command.redo()

        self.assertEqual(self.component.position, [25, 25])
        self.assertEqual(self.component.bounds, [150, 150])
        self.assertTrue(self.component.request_redraw.called)

    def test_merge(self):
        command = self.command
        command.mergeable = True
        other_command = ResizeCommand(
            component=self.component,
            data=(0, 0, 200, 200),
            previous_rectangle=(50, 50, 100, 100),
        )

        with self.assertTraitChanges(command, "data"):
            merged = command.merge(other_command)

        self.assertTrue(merged)
        self.assertEqual(command.data, (0, 0, 200, 200))
        self.assertFalse(command.mergeable)

    def test_merge_other_mergeable(self):
        command = self.command
        command.mergeable = True
        other_command = ResizeCommand(
            component=self.component,
            mergeable=True,
            data=(0, 0, 200, 200),
            previous_rectangle=(50, 50, 100, 100),
        )

        with self.assertTraitChanges(command, "data"):
            merged = command.merge(other_command)

        self.assertTrue(merged)
        self.assertEqual(command.data, (0, 0, 200, 200))
        self.assertTrue(command.mergeable)

    def test_merge_unmergeable(self):
        command = self.command
        other_command = ResizeCommand(
            component=self.component,
            data=(0, 0, 200, 200),
            previous_rectangle=(50, 50, 100, 100),
        )

        with self.assertTraitDoesNotChange(command, "data"):
            merged = command.merge(other_command)

        self.assertFalse(merged)
        self.assertEqual(command.data, (25, 25, 150, 150))

    def test_merge_wrong_component(self):
        command = self.command
        command.mergeable = True
        other_component = Component()
        other_command = ResizeCommand(
            component=other_component,
            data=(0, 0, 200, 200),
            previous_rectangle=(50, 50, 100, 100),
        )

        with self.assertTraitDoesNotChange(command, "data"):
            merged = command.merge(other_command)

        self.assertFalse(merged)
        self.assertEqual(command.data, (25, 25, 150, 150))

    def test_merge_wrong_class(self):
        command = self.command
        command.mergeable = True
        other_command = ComponentCommand(component=self.component)

        with self.assertTraitDoesNotChange(command, "data"):
            merged = command.merge(other_command)

        self.assertFalse(merged)
        self.assertEqual(command.data, (25, 25, 150, 150))


class MoveCommandTest(unittest.TestCase, EnableTestAssistant, UnittestTools):
    def setUp(self):
        self.component = Component(position=[50, 50], bounds=[100, 100])
        self.component.request_redraw = MagicMock()
        self.command = MoveCommand(self.component, (25, 25), (50, 50))

    def test_name_default(self):
        self.assertEqual(self.command.name, "Move Component")

    def test_name_alternate_component_name(self):
        self.command.component_name = "My Component"
        self.assertEqual(self.command.name, "Move My Component")

    def test_do(self):
        command = self.command

        with self.assertTraitChanges(self.component, "position", count=1):
            command.do()

        self.assertEqual(self.component.position, [25, 25])
        self.assertTrue(self.component.request_redraw.called)

    def test_do_no_previous_position(self):
        command = MoveCommand(self.component, (25, 25))
        self.assertEqual(command.previous_position, (50, 50))

    def test_do_no_new_position(self):
        with self.assertRaises(TypeError):
            MoveCommand(self.component, previous_position=(50, 50))

    def test_do_no_new_position_with_data(self):
        command = MoveCommand(
            self.component, previous_position=(50, 50), data=(25, 25)
        )
        self.assertEqual(command.data, (25, 25))

    def test_do_no_new_position_no_previous_position_with_data(self):
        command = MoveCommand(self.component, data=(25, 25))
        self.assertEqual(command.data, (25, 25))
        self.assertEqual(command.previous_position, (50, 50))

    def test_undo(self):
        command = self.command
        command.do()

        with self.assertTraitChanges(self.component, "position", count=1):
            command.undo()

        self.assertEqual(self.component.position, [50, 50])
        self.assertTrue(self.component.request_redraw.called)

    def test_redo(self):
        command = self.command
        command.do()
        command.undo()

        with self.assertTraitChanges(self.component, "position", count=1):
            command.redo()

        self.assertEqual(self.component.position, [25, 25])
        self.assertTrue(self.component.request_redraw.called)

    def test_merge(self):
        command = self.command
        command.mergeable = True
        other_command = MoveCommand(
            component=self.component,
            data=(0, 0),
            previous_position=(50, 50),
        )

        with self.assertTraitChanges(command, "data"):
            merged = command.merge(other_command)

        self.assertTrue(merged)
        self.assertEqual(command.data, (0, 0))
        self.assertFalse(command.mergeable)

    def test_merge_other_mergeable(self):
        command = self.command
        command.mergeable = True
        other_command = MoveCommand(
            component=self.component,
            mergeable=True,
            data=(0, 0),
            previous_position=(50, 50),
        )

        with self.assertTraitChanges(command, "data"):
            merged = command.merge(other_command)

        self.assertTrue(merged)
        self.assertEqual(command.data, (0, 0))
        self.assertTrue(command.mergeable)

    def test_merge_unmergeable(self):
        command = self.command
        other_command = MoveCommand(
            component=self.component, data=(0, 0), previous_position=(50, 50)
        )

        with self.assertTraitDoesNotChange(command, "data"):
            merged = command.merge(other_command)

        self.assertFalse(merged)
        self.assertEqual(command.data, (25, 25))

    def test_merge_wrong_component(self):
        command = self.command
        command.mergeable = True
        other_component = Component()
        other_command = MoveCommand(
            component=other_component,
            data=(0, 0),
            previous_position=(50, 50),
        )

        with self.assertTraitDoesNotChange(command, "data"):
            merged = command.merge(other_command)

        self.assertFalse(merged)
        self.assertEqual(command.data, (25, 25))

    def test_merge_wrong_class(self):
        command = self.command
        command.mergeable = True
        other_command = ComponentCommand(component=self.component)

        with self.assertTraitDoesNotChange(command, "data"):
            merged = command.merge(other_command)

        self.assertFalse(merged)
        self.assertEqual(command.data, (25, 25))
