#
# (C) Copyright 2015 Enthought, Inc., Austin, TX
# All right reserved.
#
# This file is open source software distributed according to the terms in
# LICENSE.txt
#

"""
Command Tools
=============

This module provides classes for tools that work with Apptools' Undo/Redo
Command stack.

"""

from __future__ import (division, absolute_import, print_function,
                        unicode_literals)

from apptools.undo.api import ICommandStack, IUndoManager
from traits.api import Callable, Instance

from enable.base_tool import BaseTool


class BaseCommandTool(BaseTool):
    """ A tool which can push commands onto a command stack

    This is a base class for all tools that want to be able to issue
    undoable commands.

    """

    # The command that the tool creates in response to user action.
    command = Callable

    # The command stack to push to.
    command_stack = Instance(ICommandStack)


class BaseUndoTool(BaseCommandTool):
    """ A tool with access to an UndoManager

    This is a base class for all tools that want to be able to access undo and
    redo functionality.

    """

    # The undo manager
    undo_manager = Instance(IUndoManager)

    def undo(self):
        """ Call undo on the UndoManager """
        self.undo_manager.undo()

    def redo(self):
        """ Call redo on the UndoManager """
        self.undo_manager.redo()
