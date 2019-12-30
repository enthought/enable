from traits.api import HasTraits, Instance
from traitsui.api import Item, View

from enable.api import Container, TextField, Window, Component, ComponentEditor

size = (500, 400)


class Demo(HasTraits):
    canvas = Instance(Component)

    traits_view = View(Item('canvas', editor=ComponentEditor(),
                            show_label=False, width=200, height=200),
                       resizable=True)

    def _canvas_default(self):
        text_field = TextField(position=[25, 100], width=200)

        text = "This a test with a text field\nthat has more text than\n"
        text += "can fit in it."
        text_field2 = TextField(position=[25, 200], width=200,
                                height=50, multiline=True,
                                text=text, font="Courier New 14")

        text_field3 = TextField(position=[250, 50], height=300,
                                width=200, multiline=True,
                                font="Courier New 14")

        container = Container(bounds=size, bgcolor='grey')
        container.add(text_field, text_field2, text_field3)
        return container


if __name__ == '__main__':
    Demo().configure_traits()
