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
Enable Commands
===============

This module provides :py:class:`pyface.undo.abstract_command.AbstractCommand`
subclasses for common component manipulations, such as moving, resizing and
setting attribute values.

"""

from pyface.undo.api import AbstractCommand
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

    def __init__(self, component, **traits):
        super(ComponentCommand, self).__init__(component=component, **traits)

    # -------------------------------------------------------------------------
    # traits handlers
    # -------------------------------------------------------------------------

    def _component_name_default(self):
        if self.component is not None:
            return camel_case_to_words(self.component.__class__.__name__)
        return ""


class ResizeCommand(ComponentCommand):
    """ Command for resizing a component

    This handles the logic of moving a component and merging successive moves.
    This class provides  ``_change_rectangle`` and ``_merge_data`` methods that
    subclasses can override to change the reszie behaviour in a uniform way for
    undo and redo operations.

    Parameters
    ----------

    component : Component instance
        The component being moved.

    new_rectangle : tuple of (x, y, w, h)
        The rectangle representing the new position and bounds.

    previous_rectangle : tuple of (x, y, w, h)
        The rectangle representing the previous position and bounds.

    **traits :
        Any other trait values that need to be passed in at creation time.

    The ``new_rectangle`` argument is the same as the ``data`` trait on the
    class.  If both are provided, the ``new_rectangle`` value is used.

    """

    #: The new rectangle of the component as a tuple (x, y, width, height).
    data = Tuple

    #: The old rectangle of the component as a tuple (x, y, width, height).
    previous_rectangle = Tuple

    #: whether additional resizes can be merged or if the resize is finished.
    mergeable = Bool

    def __init__(self, component, new_rectangle=None, previous_rectangle=None,
                 **traits):
        if previous_rectangle is None:
            previous_rectangle = tuple(component.position) + tuple(
                component.bounds
            )

        if new_rectangle is None:
            if "data" in traits:
                data = traits.pop("data")
            else:
                raise TypeError(
                    "ResizeCommand __init__ method requires "
                    "'new_rectangle' argument."
                )
        else:
            data = new_rectangle

        super(ResizeCommand, self).__init__(
            component=component,
            data=data,
            previous_rectangle=previous_rectangle,
            **traits,
        )

    @classmethod
    def move_command(cls, component, new_position, previous_position=None,
                     **traits):
        """ Factory that creates a ResizeCommand implementing a move operation

        This allows a MoveTool to create move commands that can be easily
        merged with resize commands.

        """
        bounds = tuple(component.bounds)
        new_rectangle = new_position + bounds
        if previous_position is not None:
            previous_rectangle = previous_position + bounds
        else:
            previous_rectangle = None
        return cls(
            component=component,
            new_rectangle=new_rectangle,
            previous_rectangle=previous_rectangle,
            **traits,
        )

    # -------------------------------------------------------------------------
    # AbstractCommand interface
    # -------------------------------------------------------------------------

    def merge(self, other):
        if (self.mergeable
                and isinstance(other, self.__class__)
                and other.component == self.component):
            return self._merge_data(other)
        return super(ResizeCommand, self).merge(other)

    def do(self):
        if self.previous_rectangle == ():
            self.previous_rectangle = tuple(
                self.component.position + self.component.bounds
            )
        self.redo()

    def redo(self):
        self._change_rectangle(self.data)

    def undo(self):
        self._change_rectangle(self.previous_rectangle)

    # -------------------------------------------------------------------------
    # Private interface
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # traits handlers
    # -------------------------------------------------------------------------

    def _name_default(self):
        return "Resize " + self.component_name


class MoveCommand(ComponentCommand):
    """ A command that moves a component

    This handles the logic of moving a component and merging successive moves.
    This class provides  ``_change_position`` and ``_merge_data`` methods that
    subclasses can override to change the move behaviour in a uniform way for
    undo and redo operations.

    Parameters
    ----------

    component : Component instance
        The component being moved.

    new_position : tuple of (x, y)
        The tuple representing the new position.

    previous_position : tuple of (x, y)
        The tuple representing the previous position.

    **traits :
        Any other trait values that need to be passed in at creation time.

    The ``new_position`` argument is the same as the ``data`` trait on the
    class.  If both are provided, the ``new_position`` value is used.
    """

    #: The new position of the component as a tuple (x, y).
    data = Tuple

    #: The old position of the component as a tuple (x, y).
    previous_position = Tuple

    #: whether additional moves can be merged or if the move is finished.
    mergeable = Bool

    def __init__(self, component, new_position=None, previous_position=None,
                 **traits):
        if previous_position is None:
            previous_position = component.position

        if new_position is None:
            if "data" in traits:
                data = traits.pop("data")
            else:
                raise TypeError(
                    "MoveCommand __init__ method requires "
                    "'new_position' argument."
                )
        else:
            data = new_position

        super(MoveCommand, self).__init__(
            component=component,
            data=data,
            previous_position=previous_position,
            **traits,
        )

    # -------------------------------------------------------------------------
    # AbstractCommand interface
    # -------------------------------------------------------------------------

    def do(self):
        self.redo()

    def redo(self):
        self._change_position(self.data)

    def undo(self):
        self._change_position(self.previous_position)

    def merge(self, other):
        if (self.mergeable
                and isinstance(other, self.__class__)
                and other.component == self.component):
            return self._merge_data(other)
        return super(MoveCommand, self).merge(other)

    # -------------------------------------------------------------------------
    # Private interface
    # -------------------------------------------------------------------------

    def _change_position(self, position):
        self.component.position = list(position)
        self.component._layout_needed = True
        self.component.request_redraw()

    def _merge_data(self, other):
        self.data = other.data
        self.mergeable = other.mergeable
        return True

    # -------------------------------------------------------------------------
    # traits handlers
    # -------------------------------------------------------------------------

    def _name_default(self):
        return "Move " + self.component_name
