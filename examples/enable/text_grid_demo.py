from numpy import array
from traits.api import HasTraits, Instance
from traitsui.api import Item, View

from enable.api import Container, Window, Component, ComponentEditor
from enable.text_grid import TextGrid

size = (400, 100)


class Demo(HasTraits):
    canvas = Instance(Component)

    traits_view = View(Item('canvas', editor=ComponentEditor(),
                            show_label=False, width=200, height=200),
                       resizable=True)

    def _canvas_default(self):
        strings = array([["apple", "banana", "cherry", "durian"],
                         ["eggfruit", "fig", "grape", "honeydew"]])
        grid = TextGrid(string_array=strings)
        container = Container(bounds=size)
        container.add(grid)
        return container


if __name__ == "__main__":
    Demo().configure_traits()
