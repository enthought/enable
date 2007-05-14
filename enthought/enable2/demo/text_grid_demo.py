
from numpy import array


from enthought.enable2.text_grid import TextGrid
from enthought.enable2.wx_backend.api import Window
from enthought.enable2.api import Container


from demo_base import DemoFrame, demo_main

class MyFrame(DemoFrame):
    def _create_window(self):
        
        strings = array([["apple", "banana", "cherry", "durian"],
                         ["eggfruit", "fig", "grape", "huckleberry"]])
        grid = TextGrid(string_array=strings)
        container = Container(bounds=[500,100])
        container.add(grid)
        return Window(self, -1, component=container)

if __name__ == "__main__":
    demo_main(MyFrame, size=[500,100])
    
