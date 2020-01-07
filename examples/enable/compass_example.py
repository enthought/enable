from __future__ import print_function

from enable.api import OverlayContainer, Compass, Window
from enable.example_support import demo_main, DemoFrame


class Demo(DemoFrame):

    def _create_component(self):
        compass = Compass(scale=2, color="blue", clicked_color="red")

        container = OverlayContainer()
        container.add(compass)

        compass.on_trait_change(self._arrow_printer, "clicked")
        self.compass = compass
        return container

    def _create_window(self):
        return Window(self, -1, component=self._create_component())

    def _arrow_printer(self):
        print("Clicked:", self.compass.clicked)


if __name__ == "__main__":
    # Save demo so that it doesn't get garbage collected when run within
    # existing event loop (i.e. from ipython).
    demo = demo_main(Demo, title="Slider example")
