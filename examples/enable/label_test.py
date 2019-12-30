""" Small demo of the Label component.  """

from traits.api import HasTraits, Instance
from traitsui.api import Item, View

from enable.api import Component, ComponentEditor
from enable.label import Label


class Demo(HasTraits):
    canvas = Instance(Component)

    traits_view = View(Item('canvas', editor=ComponentEditor(),
                            show_label=False, width=200, height=200),
                       resizable=True)

    def _canvas_default(self):
        label = Label(bounds=[100, 50], position=[50, 50], text="HELLO")
        label.bgcolor = "lightpink"
        return label


if __name__ == "__main__":
    Demo().configure_traits()
