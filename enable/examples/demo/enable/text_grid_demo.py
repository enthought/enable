from numpy import array

from enable.api import Container
from enable.example_support import DemoFrame, demo_main
from enable.text_grid import TextGrid

size = (400, 100)


class Demo(DemoFrame):
    def _create_component(self):
        strings = array([["apple", "banana", "cherry", "durian"],
                         ["eggfruit", "fig", "grape", "honeydew"]])
        grid = TextGrid(string_array=strings)
        container = Container(bounds=size)
        container.add(grid)
        return container

if __name__ == "__main__":
    # Save demo so that it doesn't get garbage collected when run within
    # existing event loop (i.e. from ipython).
    demo = demo_main(Demo, size=size)
