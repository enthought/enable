# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
from enable.api import OverlayContainer, Slider
from enable.example_support import demo_main, DemoFrame


class Demo(DemoFrame):
    def _create_component(self):
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
    # Save demo so that it doesn't get garbage collected when run within
    # existing event loop (i.e. from ipython).
    demo = demo_main(Demo, title="Slider example")
