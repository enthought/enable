#
# (C) Copyright 2015 Enthought, Inc., Austin, TX
# All right reserved.
#
# This file is open source software distributed according to the terms in
# LICENSE.txt
#

"""
Undoable Move Tool
==================

This example shows how to integrate a simple component move tool with apptools
undo/redo infrastructure.

"""
from __future__ import (division, absolute_import, print_function,
                        unicode_literals)

from apptools.undo.api import (CommandStack, ICommandStack, IUndoManager,
                               UndoManager)
from apptools.undo.action.api import UndoAction, RedoAction
from pyface.action.api import Action, Group, MenuBarManager, MenuManager
from traits.api import Instance

from enable.api import Container, Window, KeySpec
from enable.example_application import DemoApplication, demo_main
from enable.primitives.api import Box
from enable.tools.apptools.api import MoveCommandTool, UndoTool


class UndoableMoveApplication(DemoApplication):
    """ Example of using a MoveCommandTool with undo/redo support.

    You can use left/right arrow keys to step through the move history, or
    use the standard undo/redo menu items.

    """

    #: The apptools undo manager the application uses.
    undo_manager = Instance(IUndoManager)

    #: The command stack that the MoveCommandTool will use.
    command_stack = Instance(ICommandStack)

    #-------------------------------------------------------------------------
    # DemoApplication interface
    #-------------------------------------------------------------------------

    def _create_window(self):
        box = Box(bounds=[100,100], position=[50,50], color='red')

        move_tool = MoveCommandTool(component=box,
                                    command_stack=self.command_stack)
        box.tools.append(move_tool)

        container = Container(bounds=[600, 600])
        container.add(box)

        undo_tool = UndoTool(component=container,
                             undo_manager=self.undo_manager,
                             undo_keys=[KeySpec('Left')],
                             redo_keys=[KeySpec('Right')])
        container.tools.append(undo_tool)

        window = Window(self.control, -1, component=container)
        return window

    #-------------------------------------------------------------------------
    # Traits handlers
    #-------------------------------------------------------------------------

    def _menu_bar_manager_default(self):
        # Create an action that exits the application.
        exit_action = Action(name='E&xit', on_perform=self.close)
        self.exit_action = exit_action
        file_menu = MenuManager(name='&File')
        file_menu.append(Group(exit_action))

        self.undo = UndoAction(undo_manager=self.undo_manager,
                               accelerator='Ctrl+Z')
        self.redo = RedoAction(undo_manager=self.undo_manager,
                               accelerator='Ctrl+Shift+Z')
        menu_bar_manager = MenuBarManager(
            file_menu,
            MenuManager(
                self.undo,
                self.redo,
                name='&Edit')
        )
        return menu_bar_manager

    def _undo_manager_default(self):
        return UndoManager()

    def _command_stack_default(self):
        stack = CommandStack(undo_manager=self.undo_manager)
        self.undo_manager.active_stack = stack
        return stack


if __name__ == "__main__":
    # Save demo so that it doesn't get garbage collected when run within
    # existing event loop (i.e. from ipython).
    demo_main(UndoableMoveApplication, size=(600, 600))
