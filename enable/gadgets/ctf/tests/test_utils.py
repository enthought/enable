
from enable.api import Component
from enable.gadgets.ctf.piecewise import PiecewiseFunction
from enable.gadgets.ctf.utils import (AlphaFunctionUIAdapter,
                                      ColorFunctionUIAdapter)


def build_piecewise_func(data):
    pf = PiecewiseFunction(key=lambda x: x[0])

    for value in data:
        pf.insert(value)

    return pf


def test_alpha_function_ui_adapter():
    func_data = [(0.0, 0.5), (0.5, 0.66), (1.0, 1.0)]
    function = build_piecewise_func(func_data)
    component = Component(bounds=(100.0, 100.0))
    adapter = AlphaFunctionUIAdapter(component=component, function=function)
    assert adapter.function_index_at_position(0, 50) == 0
    assert adapter.function_index_at_position(0, 0) is None


def test_color_function_ui_adapter():
    func_data = [(0.0, 0.5, 0.5, 0.5), (1.0, 1.0, 1.0, 1.0)]
    function = build_piecewise_func(func_data)
    component = Component(bounds=(100.0, 100.0))
    adapter = ColorFunctionUIAdapter(component=component, function=function)
    assert adapter.function_index_at_position(0, 0) == 0
    assert adapter.function_index_at_position(50, 0) is None
