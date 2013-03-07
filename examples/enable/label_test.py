""" Small demo of the Label component.  """

from enable.example_support import DemoFrame, demo_main
from enable.label import Label

from enable.api import Window


class MyFrame(DemoFrame):

    def _create_window(self):
        label = Label(bounds=[100, 50], position=[50,50], text="HELLO")
        label.bgcolor = "lightpink"
        return Window(self, -1, component=label)


if __name__ == "__main__":
    # Save demo so that it doesn't get garbage collected when run within
    # existing event loop (i.e. from ipython).
    demo = demo_main(MyFrame)
