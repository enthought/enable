"""
This demonstrates the resize tool.
"""
from traits.api import HasTraits, Instance
from traitsui.api import Item, View

from enable.api import Component, Container, ComponentEditor
from enable.tools.resize_tool import ResizeTool


class Box(Component):
    resizable = ""

    def _draw_mainlayer(self, gc, view_bounds=None, mode="default"):
        with gc:
            dx, dy = self.bounds
            x, y = self.position
            gc.set_fill_color((1.0, 0.0, 0.0, 1.0))
            gc.rect(x, y, dx, dy)
            gc.fill_path()


class Demo(HasTraits):
    canvas = Instance(Component)

    traits_view = View(Item('canvas', editor=ComponentEditor(),
                            show_label=False, width=200, height=200),
                       resizable=True)

    def _canvas_default(self):
        box = Box(bounds=[100.0, 100.0], position=[50.0, 50.0])
        box.tools.append(ResizeTool(component=box,
                                    hotspots=set(["top", "left", "right",
                                                  "bottom", "top left",
                                                  "top right", "bottom left",
                                                  "bottom right"])))
        container = Container(bounds=[500, 500])
        container.add(box)
        return container


if __name__ == "__main__":
    Demo().configure_traits()
