# ------------------------------------------------------------------------------
#  Copyright (c) 2014, Enthought, Inc.
#  All rights reserved.
# ------------------------------------------------------------------------------
from abc import ABCMeta

import six

import kiwisolver as kiwi


@six.add_metaclass(ABCMeta)
class LinearSymbolic(object):
    """ An abstract base class for testing linear symbolic interfaces.

    """


LinearSymbolic.register(kiwi.Variable)
LinearSymbolic.register(kiwi.Term)
LinearSymbolic.register(kiwi.Expression)
