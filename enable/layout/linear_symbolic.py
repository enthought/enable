#------------------------------------------------------------------------------
#  Copyright (c) 2014, Enthought, Inc.
#  All rights reserved.
#------------------------------------------------------------------------------
from abc import ABCMeta

import kiwisolver as kiwi


class LinearSymbolic(object):
    """ An abstract base class for testing linear symbolic interfaces.

    """
    __metaclass__ = ABCMeta


LinearSymbolic.register(kiwi.Variable)
LinearSymbolic.register(kiwi.Term)
LinearSymbolic.register(kiwi.Expression)
