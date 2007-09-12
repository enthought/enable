

from enthought.enable2.api import Canvas, Viewport, Window, Scrolled
from enthought.enable2.tools.api import ViewportPanTool, ViewportZoomTool
from enthought.enable2.primitives.api import Box
from enthought.enable2.example_support import demo_main, DemoFrame

class MyFrame(DemoFrame):

    def _create_window(self):

        canvas = Canvas(bgcolor="lightsteelblue", draw_axes=True)
        from basic_move import Box
        box = Box(color="red", bounds=[50,50], resizable="")
        box.position= [75,75]
        canvas.add(box)


        viewport = Viewport(component=canvas, enable_zoom=False)
        viewport.view_position = [0,0]
        viewport.tools.append(ViewportPanTool(viewport))

        scrolled = Scrolled(canvas, fit_window = True,
                            inside_padding_width = 0,
                            mousewheel_scroll = False,
                            viewport_component = viewport)

        return Window(self, -1, component=scrolled)

if __name__ == "__main__":
    demo_main(MyFrame, title="Canvas example")
