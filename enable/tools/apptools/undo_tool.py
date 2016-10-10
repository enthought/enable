#
# (C) Copyright 2015 Enthought, Inc., Austin, TX
# All right reserved.
#
# This file is open source software distributed according to the terms in
# LICENSE.txt
#
"""
UndoTool
========

Tool that triggers undo or redo when keys are pressed.

"""

from __future__ import (division, absolute_import, print_function,
                        unicode_literals)

# Enthought library imports
from traits.api import Instance, List

# Local library imports
from enable.base_tool import KeySpec
from .command_tool import BaseUndoTool


# default undo/redo/clear key specifications
ctrl_z = KeySpec('z', 'control')
ctrl_shift_z = KeySpec('z', 'control', 'shift')


class UndoTool(BaseUndoTool):
    """ Tool that triggers undo or redo when keys are pressed """

    #: the key sequences which trigger undo actions
    undo_keys = List(Instance(KeySpec), [ctrl_z])

    #: the key sequences which trigger redo actions
    redo_keys = List(Instance(KeySpec), [ctrl_shift_z])

    def normal_key_pressed(self, event):
        """ Respond to key presses which match either the undo or redo keys """
        if self.undo_manager is not None:
            for key in self.undo_keys:
                if key.match(event):
                    self.undo_manager.undo()
                    event.handled = True
                    return
            for key in self.redo_keys:
                if key.match(event):
                    self.undo_manager.redo()
                    event.handled = True
                    return
