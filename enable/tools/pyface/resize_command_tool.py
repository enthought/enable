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
ResizeCommandTool
=================

A CommandTool that uses Pyface's undo/redo infrastructure to create undoable
resize commands.
"""

from traits.api import Bool, Tuple

from enable.tools.resize_tool import ResizeTool

from .command_tool import BaseCommandTool
from .commands import ResizeCommand


class ResizeCommandTool(ResizeTool, BaseCommandTool):
    """ Resize tool which pushes ResizeCommands onto a CommandStack

    This tool pushes a single ResizeCommand onto its CommandStack at
    the end of the drag operation.  If the drag is cancelled, then no command
    is issued, and no commands are issued during the drag operation.
    """

    # -------------------------------------------------------------------------
    # 'ResizeCommandTool' interface
    # -------------------------------------------------------------------------

    #: Whether or not subsequent moves can be merged with this one.
    mergeable = Bool

    #: The initial component position.
    _initial_rectangle = Tuple(0, 0, 0, 0)

    # -------------------------------------------------------------------------
    # 'DragTool' interface
    # -------------------------------------------------------------------------

    def drag_start(self, event):
        if self.component is not None:
            # we need to save the initial position to give to the Command
            self._initial_rectangle = tuple(
                self.component.position + self.component.bounds
            )
        result = super(ResizeCommandTool, self).drag_start(event)
        return result

    def drag_end(self, event):
        """ End the drag operation, issuing a ResizeCommands
        """
        if self.component is not None:
            command = self.command(
                component=self.component,
                new_rectangle=tuple(
                    self.component.position + self.component.bounds
                ),
                previous_rectangle=self._initial_rectangle,
                final=True,
            )
            self.command_stack.push(command)
            event.handled = True
            return super(ResizeCommandTool, self).drag_end(event)
        return True

    def drag_cancel(self, event):
        """ Restore the component's position if the drag is cancelled.

        A drag is usually cancelled by receiving a mouse_leave event when
        `end_drag_on_leave` is True, or by the user pressing any of the
        `cancel_keys`.
        """
        if self.component is not None:
            self.component.position = list(self._initial_rectangle)
            event.handled = True
        return True

    # -------------------------------------------------------------------------
    # Trait handlers
    # -------------------------------------------------------------------------

    def _command_default(self):
        return ResizeCommand
