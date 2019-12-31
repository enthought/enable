from traits.api import HasTraits, Instance
from traitsui.api import Item, View

from enable.api import Canvas, Viewport, ComponentEditor, Component
from enable.tools.api import ViewportPanTool


class Demo(HasTraits):
    canvas = Instance(Component)

    traits_view = View(Item('canvas', editor=ComponentEditor(),
                            show_label=False, width=200, height=200),
                       resizable=True, title="Canvas Example")

    def _canvas_default(self):
        canvas = Canvas(bgcolor="lightsteelblue", draw_axes=True)
        from .basic_move import Box
        box = Box(color="red", bounds=[50, 50], resizable="")
        box.position = [75, 75]
        canvas.add(box)
        viewport = Viewport(component=canvas)
        viewport.view_position = [0, 0]
        viewport.tools.append(ViewportPanTool(viewport))
        return viewport


if __name__ == "__main__":
    Demo().configure_traits()
