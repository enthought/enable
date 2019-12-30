"""
This demo is a canvas that showcases some of the drawing primitives in Enable.
"""
from traits.api import HasTraits, Instance
from traitsui.api import Item, View

from enable.api import ComponentEditor, Component
from enable.drawing.api import DragLine, DragPolygon, DragSegment, \
    DrawingTool, PointLine, PointPolygon, DrawingCanvas, ToolbarButton, \
    DrawingCanvasToolbar


class ResetButton(ToolbarButton):

    def perform(self, event):
        if self.canvas and self.canvas.active_tool:
            self.canvas.active_tool.reset()
            self.canvas.request_redraw()
        return


class ActivateButton(ToolbarButton):
    tool = Instance(DrawingTool)

    def perform(self, event):
        if not self.canvas or not self.tool:
            return
        else:
            self.canvas.activate(self.tool)
            self.canvas.request_redraw()
        return

    def _tool_changed(self, old, new):
        if new:
            new.reset()
        return


class Demo(HasTraits):
    canvas = Instance(Component)

    traits_view = View(Item('canvas', editor=ComponentEditor(),
                            show_label=False, width=200, height=200),
                       resizable=True)

    def _canvas_default(self):
        canvas = DrawingCanvas(bounds=[500, 500])
        toolbar = DrawingCanvasToolbar(width=500, height=32, fit_window=False,
                                       bgcolor="lightgrey")
        canvas.toolbar = toolbar
        toolbar.canvas = canvas

        button1 = ResetButton(label="Reset", toolbar=toolbar, bounds=[50, 24])
        button2 = ActivateButton(tool=DragLine(container=canvas), label="Path",
                                 toolbar=toolbar, bounds=[50, 24])
        button3 = ActivateButton(
            tool=DragPolygon(background_color=(0, 0, 0.8, 1),
                             container=canvas),
            label="Poly", toolbar=toolbar, bounds=[50, 24])
        button4 = ActivateButton(tool=PointLine(container=canvas),
                                 label="Polyline",
                                 toolbar=toolbar, bounds=[70, 24])
        button5 = ActivateButton(tool=DragSegment(container=canvas),
                                 label="Line", toolbar=toolbar,
                                 bounds=[50, 24])
        button6 = ActivateButton(tool=PointPolygon(container=canvas),
                                 label="PointPoly", toolbar=toolbar,
                                 bounds=[80, 24])
        toolbar.add_button(button1)
        toolbar.add_button(button2)
        toolbar.add_button(button3)
        toolbar.add_button(button4)
        toolbar.add_button(button5)
        toolbar.add_button(button6)
        return canvas


if __name__ == "__main__":
    Demo().configure_traits()
