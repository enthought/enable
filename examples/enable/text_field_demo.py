from enable.api import Container, TextField, Window
from enable.example_support import DemoFrame, demo_main


size = (500, 400)


class MyFrame(DemoFrame):
    def _create_window(self):
        text_field = TextField(position=[25,100], width=200)

        text = "This a test with a text field\nthat has more text than\n"
        text += "can fit in it."
        text_field2 = TextField(position=[25,200], width=200,
                                height=50, multiline=True,
                                text=text, font="Courier New 14")

        text_field3 = TextField(position=[250,50], height=300,
                                width=200, multiline=True,
                                font="Courier New 14")

        container = Container(bounds=size, bgcolor='grey')
        container.add(text_field, text_field2, text_field3)
        return Window(self, -1, component=container)


if __name__ == '__main__':
    # Save demo so that it doesn't get garbage collected when run within
    # existing event loop (i.e. from ipython).
    demo = demo_main(MyFrame, size=size)
