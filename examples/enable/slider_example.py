from __future__ import print_function

from traits.api import HasTraits, Instance
from traitsui.api import Item, View

from enable.api import OverlayContainer, Slider, Component, \
    ComponentEditor


class Demo(HasTraits):
    canvas = Instance(Component)

    traits_view = View(Item('canvas', editor=ComponentEditor(),
                            show_label=False, width=200, height=200),
                       resizable=True)

    def _canvas_default(self):
        slider = Slider()
        slider.set_slider_pixels(10)
        slider.slider_thickness = 5
        slider.set_endcap_percent(0.1)
        slider.min = 0
        slider.max = 100
        slider.value = 40
        slider.padding = 25
        slider.slider = "cross"
        slider.orientation = "h"
        slider.num_ticks = 4
        slider.set_tick_percent(0.05)

        container = OverlayContainer()
        container.add(slider)

        slider.on_trait_change(self.val_changed, "value")
        self.slider = slider
        return container

    def val_changed(self):
        print(self.slider.value)


if __name__ == "__main__":
    Demo().configure_traits()
