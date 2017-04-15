from bisect import bisect
from types import FloatType


class PiecewiseFunction(object):
    """ A piecewise linear function.
    """
    def __init__(self, key=None):
        self.keyfunc = key or (lambda x: id(x))
        self.clear()

    def clear(self):
        self._keys = []
        self._values = []

    def insert(self, value):
        key = self.keyfunc(value)
        index = bisect(self._keys, key)
        self._keys.insert(index, key)
        self._values.insert(index, value)

    def items(self):
        return self._values

    def neighbor_indices(self, key_value):
        index = bisect(self._keys, key_value)
        try:
            value = self._keys[index]
            if key_value < value:
                return (max(0, index-1), index)
        except IndexError:
            max_index = self.size() - 1
            return (min(index, max_index), min(index+1, max_index))

    def remove(self, index):
        del self._keys[index]
        del self._values[index]

    def size(self):
        return len(self._values)

    def update(self, index, value):
        key = self.keyfunc(value)
        self._keys[index] = key
        self._values[index] = value

    def value_at(self, index):
        return self._values[index]

    def values(self):
        # Return a copy
        return self._values[:]


def verify_values(function_values):
    """Make sure a sequence of values are valid function values.
        - Function values must be sequences of length 2 or greater
        - All values in a function must have the same length
        - All subvalues in a value sequence must be floating point numbers
        between 0 and 1, inclusive.
    """
    sub_size = -1
    for value in function_values:
        try:
            if sub_size < 0:
                sub_size = len(value)
                if sub_size < 2:
                    return False
            elif len(value) != sub_size:
                return False
            for sub_val in value:
                if not isinstance(sub_val, FloatType):
                    return False
                if not (0.0 <= sub_val <= 1.0):
                    return False
        except TypeError:
            return False

    return True
