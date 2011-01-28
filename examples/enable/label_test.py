""" Small demo of the Label component.  """

from enthought.enable.example_support import DemoFrame, demo_main
from enthought.enable.label import Label

from enthought.enable.api import Component, Container, Pointer, Window

class MyFrame(DemoFrame):

    def _create_window(self):
        label = Label(bounds=[100, 50], position=[50,50], text="HELLO")
        label.bgcolor = "red"
        return Window(self, -1, component=label)

if __name__ == "__main__":
    demo_main(MyFrame, title="Click and drag to move the box")

# EOF
