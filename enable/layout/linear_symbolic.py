from abc import ABCMeta

import kiwisolver as kiwi


class LinearSymbolic(object, metaclass=ABCMeta):
    """ An abstract base class for testing linear symbolic interfaces.

    """


LinearSymbolic.register(kiwi.Variable)
LinearSymbolic.register(kiwi.Term)
LinearSymbolic.register(kiwi.Expression)
