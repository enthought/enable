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


class ResizeCommand(ComponentCommand):
    """ Command for resizing a component

    This handles the logic of moving a component and merging successive moves.
    This class provides  ``_change_rectangle`` and ``_merge_data`` methods that
    subclasses can override to change the move behaviour in a uniform way for
    undo and redo operations.

    """

    #: The new rectangle of the component as a tuple (x, y, width, height).
    data = Tuple

    #: The old rectangle of the component as a tuple (x, y, width, height).
    previous_rectangle = Tuple

    #: whether additional resizes can be merged or if the resize is finished.
    mergeable = Bool

    @classmethod
    def move_command(cls, component, data, previous_position, **traits):
        """ Factory that creates a ResizeCommand implementing a move operation

        This allows a MoveTool to create move commands that can be easily
        merged with resize commands.

        """
        bounds = component.bounds
        data += tuple(bounds)
        previous_rectangle = previous_position + tuple(bounds)
        return cls(
            component=component,
            data=data,
            previous_rectangle=previous_rectangle
            **traits)


    #-------------------------------------------------------------------------
    # AbstractCommand interface
    #-------------------------------------------------------------------------

    def merge(self, other):
        if self.mergeable and isinstance(other, self.__class__) and \
                other.component == self.component:
            return self._merge_data(other)
        return super(ResizeCommand, self).merge(other)

    def do(self):
        if self.previous_rectangle == ():
            self.previous_rectangle = tuple(self.component.position +
                                            self.component.bounds)
        self.redo()

    def redo(self):
        self._change_rectangle(self.data)

    def undo(self):
        self._change_rectangle(self.previous_rectangle)

    #-------------------------------------------------------------------------
    # Private interface
    #-------------------------------------------------------------------------

    def _change_rectangle(self, rectangle):
        x, y, w, h = rectangle
        self.component.position = [x, y]
        self.component.bounds = [w, h]
        self.component._layout_needed = True
        self.component.request_redraw()

    def _merge_data(self, other):
        self.data = other.data
        self.mergeable = other.mergeable
        return True

    #-------------------------------------------------------------------------
    # traits handlers
    #-------------------------------------------------------------------------

    def _name_default(self):
        return "Resize "+self.component_name


class MoveCommand(ComponentCommand):
    """ A command that moves a component

    This handles the logic of moving a component and merging successive moves.
    This class provides  ``_change_position`` and ``_merge_data`` methods that
    subclasses can override to change the move behaviour in a uniform way for
    undo and redo operations.

    """

    #: The new position of the component as a tuple (x, y).
    data = Tuple

    #: The old position of the component as a tuple (x, y).
    previous_position = Tuple

    #: whether additional moves can be merged or if the move is finished.
    mergeable = Bool

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

    def merge(self, other):
        if self.mergeable and isinstance(other, self.__class__) and \
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
        self.data = other.data
        self.mergeable = other.mergeable
        return True

    #-------------------------------------------------------------------------
    # traits handlers
    #-------------------------------------------------------------------------

    def _name_default(self):
        return "Move "+self.component_name
