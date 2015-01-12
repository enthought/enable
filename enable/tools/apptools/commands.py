#
# (C) Copyright 2015 Enthought, Inc., Austin, TX
# All right reserved.
#
# This file is open source software distributed according to the terms in
# LICENSE.txt
#

"""
Enable Commands
===============

This module provides :py:class:`apptools.undo.abstract_command.AbstractCommand`
subclasses for common component manipulations, such as moving, resizing and
setting attribute values.

"""

from __future__ import (division, absolute_import, print_function,
                        unicode_literals)

from apptools.undo.api import AbstractCommand
from traits.api import Bool, Instance, Tuple, Unicode
from traits.util.camel_case import camel_case_to_words

from enable.component import Component

class ComponentCommand(AbstractCommand):
    """ Abstract command which operates on a Component """

    #: The component the command is being performed on.
    component = Instance(Component)

    #: An appropriate name for the component that can be used by the command.
    #: The default is the class name, split into words.
    component_name = Unicode

    #-------------------------------------------------------------------------
    # traits handlers
    #-------------------------------------------------------------------------

    def _component_name_default(self):
        if self.component is not None:
            return camel_case_to_words(self.component.__class__.__name__)
        return ''


class MoveCommand(ComponentCommand):
    """ A command that moves a component

    This handles some of the logic of moving a component and merging successive
    moves.  Subclasses should call `_change_position()` when they wish to move
    the object, and should override the implementation of `_merge_data()` if
    they wish to be able to merge non-finalized moves.

    """

    #: whether the move is finished, or if additional moves can be merged.
    final = Bool

    #-------------------------------------------------------------------------
    # AbstractCommand interface
    #-------------------------------------------------------------------------

    def merge(self, other):
        if not self.final and isinstance(other, self.__class__) and \
                other.component == self.component:
            return self._merge_data(other)
        return super(MoveCommand, self).merge(other)

    #-------------------------------------------------------------------------
    # Private interface
    #-------------------------------------------------------------------------

    def _change_position(self, position):
        self.component.position = list(position)
        self.component._layout_needed = True
        self.component.request_redraw()

    def _merge_data(self, other):
        return False

    #-------------------------------------------------------------------------
    # traits handlers
    #-------------------------------------------------------------------------

    def _name_default(self):
        return "Move "+self.component_name


class MoveDeltaCommand(MoveCommand):
    """ Command that records fine-grained movement of an object

    This is suitable for being used for building up a Command from many
    incremental steps.

    """

    #: The change in position of the component as a tuple (dx, dy).
    data = Tuple

    #-------------------------------------------------------------------------
    # AbstractCommand interface
    #-------------------------------------------------------------------------

    def do(self):
        self.redo()

    def redo(self):
        x = self.component.position[0] + self.delta[0]
        y = self.component.position[1] + self.delta[1]
        self._change_position((x, y))

    def undo(self):
        x = self.component.position[0] - self.delta[0]
        y = self.component.position[1] - self.delta[1]
        self._change_position((x, y))

    #-------------------------------------------------------------------------
    # Private interface
    #-------------------------------------------------------------------------

    def _merge_data(self, other):
        x = self.data[0] + other.data[0]
        y = self.data[1] + other.data[1]
        self.data = (x, y)


class MovePositionCommand(MoveCommand):
    """ Command that records gross movement of an object """

    #: The new position of the component as a tuple (x, y).
    data = Tuple

    #: The old position of the component as a tuple (x, y).
    previous_position = Tuple

    #-------------------------------------------------------------------------
    # AbstractCommand interface
    #-------------------------------------------------------------------------

    def do(self):
        if self.previous_position == ():
            self.previous_position = tuple(self.component.position)
        self.redo()

    def redo(self):
        self._change_position(self.data)

    def undo(self):
        self._change_position(self.previous_position)

    #-------------------------------------------------------------------------
    # Private interface
    #-------------------------------------------------------------------------

    def _merge_data(self, other):
        self.data = other.data
