from traits.api import HasTraits, Instance
from traitsui.api import Item, View

from enable.api import Container, TextField, ComponentEditor, Component
from enable.stacked_container import VStackedContainer, HStackedContainer

size = (240, 240)


class Demo(HasTraits):
    canvas = Instance(Component)

    traits_view = View(Item('canvas', editor=ComponentEditor(),
                            show_label=False, width=200, height=200),
                       resizable=True)

    def _canvas_default(self):
        stack = VStackedContainer(position=[0, 0],
                                  halign='center', valign='center',
                                  # border_visible=True,
                                  fit_components='hv', auto_size=True,
                                  stack_order='top_to_bottom',
                                  bgcolor='red')

        strings = ["apple", "banana", "cherry", "durian",
                   "eggfruit", "fig", "grape", "honeydew"]

        for i, s in enumerate(strings):
            label = TextField(text=s, resizable='', bounds=[100 + i * 10, 20],
                              bgcolor='red',  # border_visible=True,
                              text_offset=1)
            number = TextField(text=str(i + 1), resizable='',
                               bgcolor='blue',  # border_visible=True,
                               text_offset=1, can_edit=False, bounds=[20, 20])
            row = HStackedContainer(fit_components='hv', auto_size=True,
                                    resizable='',
                                    valign='top', border_visible=True)
            row.add(number)
            row.add(label)
            stack.add(row)

        container = Container(position=[20, 20], bounds=size)
        container.add(stack)
        container2 = Container(bounds=size)
        container2.add(container)
        return container2


if __name__ == "__main__":
    Demo().configure_traits()
