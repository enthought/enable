#------------------------------------------------------------------------------
#  Copyright (c) 2013, Enthought, Inc.
#  All rights reserved.
#------------------------------------------------------------------------------


STRENGTHS = set(['required', 'strong', 'medium', 'weak'])


def add_symbolic_constraints(namespace):
    """ Add constraints to a namespace that are LinearExpressions of basic
    constraints.

    """
    bottom = namespace.bottom
    left = namespace.left
    width = namespace.layout_width
    height = namespace.layout_height

    namespace.right = left + width
    namespace.top = bottom + height
    namespace.h_center = left + width / 2.0
    namespace.v_center = bottom + height / 2.0


def add_symbolic_contents_constraints(namespace):
    """ Add constraints to a namespace that are LinearExpressions of basic
    constraints.

    """
    left = namespace.contents_left
    right = namespace.contents_right
    top = namespace.contents_top
    bottom = namespace.contents_bottom

    namespace.contents_width = right - left
    namespace.contents_height = top - bottom
    namespace.contents_v_center = bottom + namespace.contents_height / 2.0
    namespace.contents_h_center = left + namespace.contents_width / 2.0


def get_from_constraints_namespace(self, name):
    """ Property getter for all attributes that come from the constraints
    namespace.

    """
    return getattr(self._constraints_vars, name)
