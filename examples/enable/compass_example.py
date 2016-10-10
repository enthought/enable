from enable.api import OverlayContainer, Compass, Window
from enable.example_support import demo_main, DemoFrame


class MyFrame(DemoFrame):

    def _create_window(self):
        compass = Compass(scale=2, color="blue", clicked_color="red")

        container = OverlayContainer()
        container.add(compass)

        compass.on_trait_change(self._arrow_printer, "clicked")
        self.compass = compass
        return Window(self, component=container)

    def _arrow_printer(self):
        print("Clicked:", self.compass.clicked)


if __name__ == "__main__":
    # Save demo so that it doesn't get garbage collected when run within
    # existing event loop (i.e. from ipython).
    demo = demo_main(MyFrame, title="Slider example")
