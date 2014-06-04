"""
This demonstrates the resize tool.
"""
from enable.example_support import DemoFrame, demo_main
from enable.api import Component, Container, Window
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


class MyFrame(DemoFrame):

    def _create_window(self):
        box = Box(bounds=[100.0, 100.0], position=[50.0, 50.0])
        box.tools.append(ResizeTool(component=box,
                                    hotspots=set(["top", "left", "right",
                                                  "bottom", "top left",
                                                  "top right", "bottom left",
                                                  "bottom right"])))
        container = Container(bounds=[500, 500])
        container.add(box)
        return Window(self, -1, component=container)


if __name__ == "__main__":
    # Save demo so that it doesn't get garbage collected when run within
    # existing event loop (i.e. from ipython).
    demo = demo_main(MyFrame)
