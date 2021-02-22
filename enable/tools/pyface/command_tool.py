# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
"""
Command Tools
=============

This module provides classes for tools that work with Pyface's Undo/Redo
Command stack.

"""

from pyface.undo.api import ICommandStack, IUndoManager
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
