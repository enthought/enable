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
    width = namespace.width
    height = namespace.height

    namespace.right = left + width
    namespace.top = bottom + height
    namespace.h_center = left + width / 2.0
    namespace.v_center = bottom + height / 2.0

