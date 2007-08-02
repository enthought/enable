"""
This demonstrates the most basic drawing capabilities using Enable.  A new
component is created and added to a container.
"""


from enthought.enable2.example_support import DemoFrame, demo_main
from enthought.enable2.api import Component, Container, Window

class Box(Component):
    def _draw(self, gc, view_bounds=None, mode="default"):
        gc.save_state()
        gc.set_fill_color((1.0, 0.0, 0.0, 1.0))
        dx, dy = self.bounds
        x, y = self.position
        gc.rect(x, y, dx, dy)
        gc.fill_path()
        gc.restore_state()

class MyFrame(DemoFrame):
    def _create_window(self):
        box = Box(bounds=[100.0, 100.0], position=[50.0, 50.0])
        container = Container(bounds=[500,500])
        container.add(box)
        return Window(self, -1, component=container)

if __name__ == "__main__":
    demo_main(MyFrame)

# EOF
