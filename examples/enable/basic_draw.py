"""
This demonstrates the most basic drawing capabilities using Enable.  A new
component is created and added to a container.
"""
from __future__ import with_statement

from enthought.enable.example_support import DemoFrame, demo_main
from enthought.enable.api import Component, Container, Window

class Box(Component):

    resizable = ""

    def _draw_mainlayer(self, gc, view_bounds=None, mode="default"):
        with gc:
            gc.set_fill_color((1.0, 0.0, 0.0, 1.0))
            dx, dy = self.bounds
            x, y = self.position
            gc.rect(x, y, dx, dy)
            gc.fill_path()

class MyFrame(DemoFrame):
    def _create_window(self):
        box = Box(bounds=[100.0, 100.0], position=[50.0, 50.0])
        container = Container(bounds=[500,500])
        container.add(box)
        return Window(self, -1, component=container)

if __name__ == "__main__":
    demo_main(MyFrame)

# EOF
