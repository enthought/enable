from enable.api import Canvas, Viewport, Window
from enable.tools.api import ViewportPanTool
from enable.example_support import demo_main, DemoFrame


class MyFrame(DemoFrame):

    def _create_window(self):

        canvas = Canvas(bgcolor="lightsteelblue", draw_axes=True)
        from basic_move import Box
        box = Box(color="red", bounds=[50, 50], resizable="")
        box.position= [75, 75]
        canvas.add(box)


        viewport = Viewport(component=canvas)
        viewport.view_position = [0, 0]
        viewport.tools.append(ViewportPanTool(viewport))

        return Window(self, -1, component=viewport)


if __name__ == "__main__":
    # Save demo so that it doesn't get garbage collected when run within
    # existing event loop (i.e. from ipython).
    demo = demo_main(MyFrame, title="Canvas example")
