from math import sqrt

import numpy as np

from enable.component import Component
from enable.gadgets.ctf.piecewise import PiecewiseFunction
from traits.api import HasTraits, Callable, Float, Instance


def build_function_to_screen(component):
    def func(pos):
        bounds = component.bounds
        return tuple([clip_to_unit(z) * size for z, size in zip(pos, bounds)])
    return func


def build_screen_to_function(component):
    def func(pos):
        bounds = component.bounds
        return tuple([clip_to_unit(z / size) for z, size in zip(pos, bounds)])
    return func


def clip_to_unit(value):
    return min(max(value, 0.0), 1.0)


def point_dist(pos0, pos1):
    diff = np.subtract(pos0, pos1)
    return sqrt(diff[0]**2 + diff[1]**2)


class FunctionUIAdapter(HasTraits):
    """ A class to handle translation between screen space and function space
    """
    # The Component where the function lives
    component = Instance(Component)

    # The function being adapted
    function = Instance(PiecewiseFunction)

    # A function which maps a point from function space to screen space
    function_to_screen = Callable

    # A function which maps a point from screen space to function space
    screen_to_function = Callable

    def _function_to_screen_default(self):
        return build_function_to_screen(self.component)

    def _screen_to_function_default(self):
        return build_screen_to_function(self.component)

    def function_index_at_position(self, x, y):
        """ Implemented by subclasses to find function nodes at the given
        mouse position. Returns None if no node is found.
        """
        raise NotImplementedError


class AlphaFunctionUIAdapter(FunctionUIAdapter):
    """ UI adapter for the alpha function
    """
    # Maximum distance from a point in screen space to be considered valid.
    valid_distance = Float(8.0)

    def function_index_at_position(self, x, y):
        mouse_pos = (x, y)
        data_pos = self.screen_to_function(mouse_pos)
        indices = self.function.neighbor_indices(data_pos[0])
        values = [self.function.value_at(i) for i in indices]
        for index, val in zip(indices, values):
            val_screen = self.function_to_screen(val)
            if point_dist(val_screen, mouse_pos) < self.valid_distance:
                return index
        return None


class ColorFunctionUIAdapter(FunctionUIAdapter):
    """ UI adapter for the color function
    """
    # Maximum distance from a point in screen space to be considered valid.
    valid_distance = Float(5.0)

    def function_index_at_position(self, x, y):
        data_x = self.screen_to_function((x, y))[0]
        indices = self.function.neighbor_indices(data_x)
        values = [self.function.value_at(i) for i in indices]
        for index, val in zip(indices, values):
            val_screen = self.function_to_screen(val)[0]
            if abs(val_screen - x) < self.valid_distance:
                return index
        return None
