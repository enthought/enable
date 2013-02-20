#------------------------------------------------------------------------------
#  Copyright (c) 2012, Enthought, Inc.
#  All rights reserved.
#------------------------------------------------------------------------------
from abc import ABCMeta


class ABConstrainable(object):
    """ An abstract base class for objects that can be laid out using
    layout helpers.

    Minimally, instances need to have `top`, `bottom`, `left`, `right`,
    `layout_width`, `layout_height`, `v_center` and `h_center` attributes
    which are `LinearSymbolic` instances.

    """
    __metaclass__ = ABCMeta

