import numpy as np

from enable.component import Component
from enable.gadgets.ctf.editor_tools import (
    AlphaFunctionEditorTool, ColorFunctionEditorTool
)
from enable.gadgets.ctf.menu_tool import FunctionMenuTool
from enable.gadgets.ctf.piecewise import PiecewiseFunction
from traits.api import Event, Instance


ALPHA_DEFAULT = ((0.0, 0.0), (1.0, 1.0))
COLOR_DEFAULT = ((0.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0, 1.0))


def create_function(values):
    fn = PiecewiseFunction(key=lambda x: x[0])
    for v in values:
        fn.insert(v)
    return fn


class CtfEditor(Component):
    """ A widget for editing transfer functions.
    """

    opacities = Instance(PiecewiseFunction)
    colors = Instance(PiecewiseFunction)

    function_updated = Event

    #------------------------------------------------------------------------
    # Public interface
    #------------------------------------------------------------------------

    def add_function_node(self, function, value):
        function.insert(value)
        self.update_function()

    def edit_function_node(self, function, index, value):
        function.update(index, value)
        self.update_function()

    def remove_function_node(self, function, index):
        if index == 0 or index == (function.size()-1):
            return False

        function.remove(index)
        self.update_function()
        return True

    def update_function(self):
        self.function_updated = True
        self.request_redraw()

    #------------------------------------------------------------------------
    # Traits initialization
    #------------------------------------------------------------------------

    def _opacities_default(self):
        return create_function(ALPHA_DEFAULT)

    def _colors_default(self):
        return create_function(COLOR_DEFAULT)

    def _tools_default(self):
        alpha = AlphaFunctionEditorTool(self, function=self.opacities)
        color = ColorFunctionEditorTool(self, function=self.colors)
        menu = FunctionMenuTool(self)
        editor_tools = [alpha, color]
        for tool in editor_tools:
            tool.on_trait_change(self.update_function, 'function_updated')
        return editor_tools + [menu]

    #------------------------------------------------------------------------
    # Drawing
    #------------------------------------------------------------------------

    def draw(self, gc, **kwargs):
        color_nodes = self.colors.items()
        alpha_nodes = self.opacities.items()

        gc.clear()
        self._draw_colors(color_nodes, gc)
        self._draw_alpha(alpha_nodes, gc)

    def _draw_alpha(self, alpha_nodes, gc):
        w, h = self.width, self.height
        points = [(w * i, h * v) for (i, v) in alpha_nodes]

        with gc:
            gc.set_line_width(1.0)
            gc.set_stroke_color((0.0, 0.0, 0.0, 1.0))
            gc.lines(points)
            gc.stroke_path()

            gc.set_line_width(2.0)
            for x, y in points:
                gc.rect(x-2, y-2, 4, 4)
            gc.stroke_path()

    def _draw_colors(self, color_nodes, gc):
        w, h = self.width, self.height
        grad_stops = np.array([(x, r, g, b, 1.0)
                               for x, r, g, b in color_nodes])

        gc.rect(0, 0, w, h)
        gc.linear_gradient(0, 0, w, 0, grad_stops, 'pad',
                           'userSpaceOnUse')
        gc.fill_path()

        with gc:
            gc.set_line_width(2.0)
            for x, r, g, b in color_nodes:
                x = x * w
                gc.set_stroke_color((1.0-r, 1.0-g, 1.0-b, 1.0))
                gc.rect(x-1, 0, 2, h)
            gc.stroke_path()
