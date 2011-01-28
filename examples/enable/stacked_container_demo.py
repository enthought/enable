

from enthought.enable.example_support import DemoFrame, demo_main

from enthought.enable.api import Container, Window, TextField
from enthought.enable.stacked_container import VStackedContainer, HStackedContainer
from enthought.enable.overlay_container import OverlayContainer

class MyFrame(DemoFrame):
    def _create_window(self):

        stack = VStackedContainer(position=[0,0], bounds=[500,500],
            halign='center', valign='center', #border_visible=True,
            fit_components='hv', auto_size=True, stack_order='top_to_bottom',
            bgcolor='red')

        strings = ["apple", "banana", "cherry", "durian",
                         "eggfruit", "fig", "grape", "honeydew"]

        for i, s in enumerate(strings):
            label = TextField(text=s, resizable='', bounds=[100+i*10,20],
                bgcolor='red', #border_visible=True,
                text_offset=1)
            number = TextField(text=str(i+1), resizable='',
                bgcolor='blue', #border_visible=True,
                text_offset=1, can_edit=False, bounds=[20,20])
            row = HStackedContainer(fit_components='hv', auto_size=True,
                resizable='',
                valign='top', border_visible=True)
            row.add(number)
            row.add(label)
            stack.add(row)

        #print stack.components
        container = Container(position=[20,20], bounds=[500,500])
        container.add(stack)
        container2 = Container(bounds=[600,600])
        container2.add(container)
        return Window(self, -1, component=container2)

if __name__ == "__main__":
    demo_main(MyFrame, size=[600,600])

