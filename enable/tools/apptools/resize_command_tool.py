#
# (C) Copyright 2015 Enthought, Inc., Austin, TX
# All right reserved.
#
# This file is open source software distributed according to the terms in
# LICENSE.txt
#
"""
MoveCommandTool
===============

A MoveTool that uses AppTools' undo/redo infrastructure to create undoable move
commands.

"""

from __future__ import (division, absolute_import, print_function,
                        unicode_literals)

from traits.api import Tuple

from enable.tools.resize_tool import ResizeTool

from .command_tool import BaseCommandTool
from .commands import ResizeCommand


class ResizeCommandTool(ResizeTool, BaseCommandTool):
    """ Resize tool which pushes ResizeCommands onto a CommandStack

    This tool pushes a single MovePositionCommand onto its CommandStack at
    the end of the drag operation.  If the drag is cancelled, then no command
    is issued, and no commands are issued during the drag operation.

    """

    #: The initial component position.
    _initial_rectangle = Tuple(0, 0, 0, 0)

    def drag_start(self, event):
        if self.component is not None:
            # we need to save the initial position to give to the Command
            self._initial_rectangle = tuple(self.component.position +
                                            self.component.bounds)
        result = super(ResizeCommandTool, self).drag_start(event)
        return result


    def drag_end(self, event):
        """ End the drag operation, issuing a MovePositionCommand """
        if self.component is not None:
            command = ResizeCommand(
                component=self.component,
                data=tuple(self.component.position + self.component.bounds),
                previous_rectangle=self._initial_rectangle,
                final=True)
            self.command_stack.push(command)
            event.handled = True
            super(ResizeCommandTool, self).drag_end(event)
        return

    def drag_cancel(self, event):
        """ Restore the component's position if the drag is cancelled.

        A drag is usually cancelled by receiving a mouse_leave event when
        `end_drag_on_leave` is True, or by the user pressing any of the
        `cancel_keys`.

        """
        if self.component is not None:
            self.component.position = list(self._initial_rectangle)
            event.handled = True
        return
