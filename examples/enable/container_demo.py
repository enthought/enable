from traits.api import Enum, Float, Int, Str, Tuple

from kiva.trait_defs.kiva_font_trait import KivaFont
from enable.api import ColorTrait, Window
from enable.tools.api import DragTool
from enable.example_support import DemoFrame, demo_main

from chaco.api import PlotComponent, AbstractOverlay, OverlayPlotContainer


class Region(PlotComponent, DragTool):

    color = ColorTrait("lightblue")
    draw_layer = "plot"
    resizable = ""
    event_states = Enum("normal", "dragging")
    _offset = Tuple

    def __init__(self, color=None, **kw):
        super(Region, self).__init__(**kw)
        if color:
            self.color = color
        if "bounds" not in kw:
            self.bounds = [100, 100]

    def _draw_plot(self, gc, view_bounds=None, mode="normal"):
        with gc:
            gc.set_fill_color(self.color_)
            gc.rect(self.x, self.y, self.width, self.height)
            gc.fill_path()

    def drag_start(self, event):
        self._offset = (event.x - self.x, event.y - self.y)
        event.handled = True

    def dragging(self, event):
        self.position = [event.x - self._offset[0],
                         event.y - self._offset[1]]
        event.handled = True
        self.request_redraw()


class Overlay(AbstractOverlay):

    text = Str
    font = KivaFont("DEFAULT 16")
    alpha = Float(0.5)
    margin = Int(8)

    def __init__(self, text="", *args, **kw):
        super(Overlay, self).__init__(*args, **kw)
        self.text = text

    def overlay(self, component, gc, view_bounds=None, mode="normal"):
        with gc:
            gc.set_font(self.font)
            twidth, theight = gc.get_text_extent(self.text)[2:]
            tx = component.x + (component.width - twidth) / 2.0
            ty = component.y + (component.height - theight) / 2.0

            # Draw a small, light rectangle representing this overlay
            gc.set_fill_color((1.0, 1.0, 1.0, self.alpha))
            gc.rect(tx-self.margin, ty-self.margin,
                         twidth+2*self.margin, theight+2*self.margin)
            gc.fill_path()

            gc.set_text_position(tx, ty)
            gc.show_text(self.text)


class PlotFrame(DemoFrame):

    def _create_window(self):
        rect1 = Region("orchid", position=[50, 50])
        rect2 = Region("cornflowerblue", position=[200, 50])
        rect1.overlays.append(Overlay("One", component=rect1))
        rect2.overlays.append(Overlay("Two", component=rect2))
        container1 = OverlayPlotContainer(bounds=[400, 400], resizable="")
        container1.add(rect1, rect2)
        container1.bgcolor = (0.60, 0.98, 0.60, 0.5) #"palegreen"

        rect3 = Region("purple", position=[50, 50])
        rect4 = Region("teal", position=[200, 50])
        rect3.overlays.append(Overlay("Three", component=rect3))
        rect4.overlays.append(Overlay("Four", component=rect4))
        container2 = OverlayPlotContainer(bounds=[400, 400], resizable="")
        container2.add(rect3, rect4)
        container2.bgcolor = "navajowhite"
        container2.position = [200, 200]

        top_container = OverlayPlotContainer()
        top_container.add(container1, container2)

        return Window(self, -1, component=top_container)


if __name__ == "__main__":
    # Save demo so that it doesn't get garbage collected when run within
    # existing event loop (i.e. from ipython).
    demo = demo_main(PlotFrame, size=(600, 600))
