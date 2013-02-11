#------------------------------------------------------------------------------
#  Copyright (c) 2013, Enthought, Inc.
#  All rights reserved.
#------------------------------------------------------------------------------

from casuarius import ConstraintVariable


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

    def __getattr__(self, name):
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
        else:
            label = '{0}|{1}|{2}'.format(self._name, self._owner, name)
            res = primitives[name] = ConstraintVariable(label)
        return res

