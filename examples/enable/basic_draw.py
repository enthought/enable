"""
This demonstrates the most basic drawing capabilities using Enable.  A new
component is created and added to a container.
"""
from traits.api import HasTraits, Instance
from traitsui.api import Item, View

from enable.api import Component, ComponentEditor


class Box(Component):

    resizable = ""

    def _draw_mainlayer(self, gc, view_bounds=None, mode="default"):
        with gc:
            gc.set_fill_color((1.0, 0.0, 0.0, 1.0))
            dx, dy = self.bounds
            x, y = self.position
            gc.rect(x, y, dx, dy)
            gc.fill_path()


class Demo(HasTraits):
    canvas = Instance(Component)

    traits_view = View(Item('canvas', editor=ComponentEditor(),
                            show_label=False, width=200, height=200),
                       resizable=True)

    def _canvas_default(self):
        box = Box(bounds=[100.0, 100.0], position=[50.0, 50.0])
        return box


if __name__ == "__main__":
    Demo().configure_traits()

