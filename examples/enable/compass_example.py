from __future__ import print_function

from traits.api import HasTraits, Instance
from traitsui.api import Item, View

from enable.api import OverlayContainer, Compass, Component, ComponentEditor


class Demo(HasTraits):
    canvas = Instance(Component)

    traits_view = View(Item('canvas', editor=ComponentEditor(),
                            show_label=False, width=200, height=200),
                       resizable=True, title="Slider Example")

    def _canvas_default(self):
        compass = Compass(scale=2, color="blue", clicked_color="red")

        container = OverlayContainer()
        container.add(compass)

        compass.on_trait_change(self._arrow_printer, "clicked")
        self.compass = compass
        return container

    def _arrow_printer(self):
        print("Clicked:", self.compass.clicked)


if __name__ == "__main__":
    Demo().configure_traits()
