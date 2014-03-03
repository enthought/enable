"""
This demonstrates the most basic drawing capabilities using Enable.  A new
component is created and added to a container.
"""
from enable.example_support import DemoFrame, demo_main
from enable.api import Component, Container, Window
from enable.tools.pyface.context_menu_tool import ContextMenuTool

from pyface.action.api import MenuManager, Action

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
    def hello(self):
        print "Hello World"

    def _create_window(self):
        box = Box(bounds=[100.0, 100.0], position=[50.0, 50.0])
        menu=MenuManager()
        menu.append(Action(name="Hello World", on_perform=self.hello))
        context_menu = ContextMenuTool(component=box, menu_manager=menu)

        box.tools.append(context_menu)
        container = Container(bounds=[500,500])
        container.add(box)
        return Window(self, -1, component=container)

if __name__ == "__main__":
    demo_main(MyFrame)

# EOF
