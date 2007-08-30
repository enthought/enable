

from enthought.enable2.api import Canvas, Viewport, Window
from enthought.enable2.primitives.api import Box
from enthought.enable2.example_support import demo_main, DemoFrame

class MyFrame(DemoFrame):

    def _create_window(self):

        canvas = Canvas(bgcolor="lightsteelblue")
        from basic_move import Box
        box = Box(color="red", bounds=[50,50], resizable="")
        box.position= [75,75]
        canvas.add(box)


        viewport = Viewport(component=canvas)
        viewport.view_position = [0,0]

        return Window(self, -1, component=viewport)

if __name__ == "__main__":
    demo_main(MyFrame, title="Canvas example")
