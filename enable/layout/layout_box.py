#------------------------------------------------------------------------------
#  Copyright (c) 2013, Enthought, Inc.
#  All rights reserved.
#------------------------------------------------------------------------------

from collections import Callable
from operator import add, mul

from casuarius import ConstraintVariable


KNOWN_CONSTRAINTS = ('left', 'right', 'top', 'bottom', 'width', 'height',
    'h_center', 'v_center')
SYMBOLIC_CONSTRAINTS = {
    'right': ['left', 'width', add],
    'top': ['bottom', 'height', add],
    'h_center': ['left', 'width', 0.5, mul, add],
    'v_center': ['bottom', 'height', 0.5, mul, add],
}


class LayoutBox(object):
    """ A class which encapsulates a layout box using casuarius
    constraint variables.

    The constraint variables are created on an as-needed basis, this
    allows components to define new constraints and build layouts
    with them, without having to specifically update this client code.

    """
    def __init__(self, name, owner):
        """ Initialize a LayoutBox.

        Parameters
        ----------
        name : str
            A name to use in the label for the constraint variables in
            this layout box.

        owner : str
            The owner id to use in the label for the constraint variables
            in this layout box.

        """
        self._name = name
        self._owner = owner
        self._primitives = {}

    def primitive(self, name):
        """ Returns a primitive casuarius constraint variable for the
        given name.

        Parameters
        ----------
        name : str
            The name of the constraint variable to return.

        """
        primitives = self._primitives
        if name in primitives:
            res = primitives[name]
        elif name in SYMBOLIC_CONSTRAINTS:
            res = primitives[name] = self._compose_symbolic(name)
        else:
            label = '{0}|{1}|{2}'.format(self._name, self._owner, name)
            res = primitives[name] = ConstraintVariable(label)
        return res

    def _compose_symbolic(self, name):
        """ Returns a casuarius constraint variable for the given symbolic
        constraint name.

        Parameters
        ----------
        name : str
            The name of the symbolic constraint variable to return.

        """
        symbolic_desc = SYMBOLIC_CONSTRAINTS[name]
        operands = []
        push = operands.append
        pop = operands.pop

        # RPN evaluation
        for part in symbolic_desc:
            if isinstance(part, Callable):
                op2 = pop()
                op1 = pop()
                push(part(op1, op2))
            elif isinstance(part, basestring):
                push(self.primitive(part))
            elif isinstance(part, (float, int, long)):
                push(part)

        assert len(operands) == 1
        return pop()

    def __getattr__(self, name):
        """ Allow the primitive dictionary to act as an extension to the
        object's namespace.
        """
        if name in KNOWN_CONSTRAINTS:
            return self.primitive(name)

        return super(LayoutBox, self).__getattr__(name)

