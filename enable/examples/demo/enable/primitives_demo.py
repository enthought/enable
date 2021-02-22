# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
"""
This demo is a canvas that showcases some of the drawing primitives in Enable.
"""
from traits.api import Instance

from enable.drawing.api import (
    DragLine, DragPolygon, DragSegment, DrawingCanvas, DrawingCanvasToolbar,
    DrawingTool, PointLine, PointPolygon, ToolbarButton
)
from enable.example_support import DemoFrame, demo_main


class ResetButton(ToolbarButton):
    def perform(self, event):
        if self.canvas and self.canvas.active_tool:
            self.canvas.active_tool.reset()
            self.canvas.request_redraw()


class ActivateButton(ToolbarButton):

    tool = Instance(DrawingTool)

    def perform(self, event):
        if not self.canvas or not self.tool:
            return
        else:
            self.canvas.activate(self.tool)
            self.canvas.request_redraw()

    def _tool_changed(self, old, new):
        if new:
            new.reset()


class Demo(DemoFrame):
    def _create_component(self):
        canvas = DrawingCanvas(bounds=[500, 500])
        toolbar = DrawingCanvasToolbar(
            width=500, height=32, fit_window=False, bgcolor="lightgrey"
        )
        canvas.toolbar = toolbar
        toolbar.canvas = canvas

        button1 = ResetButton(label="Reset", toolbar=toolbar, bounds=[50, 24])
        button2 = ActivateButton(
            tool=DragLine(container=canvas),
            label="Path",
            toolbar=toolbar,
            bounds=[50, 24],
        )
        button3 = ActivateButton(
            tool=DragPolygon(
                background_color=(0, 0, 0.8, 1), container=canvas
            ),
            label="Poly",
            toolbar=toolbar,
            bounds=[50, 24],
        )
        button4 = ActivateButton(
            tool=PointLine(container=canvas),
            label="Polyline",
            toolbar=toolbar,
            bounds=[70, 24],
        )
        button5 = ActivateButton(
            tool=DragSegment(container=canvas),
            label="Line",
            toolbar=toolbar,
            bounds=[50, 24],
        )
        button6 = ActivateButton(
            tool=PointPolygon(container=canvas),
            label="PointPoly",
            toolbar=toolbar,
            bounds=[80, 24],
        )
        toolbar.add_button(button1)
        toolbar.add_button(button2)
        toolbar.add_button(button3)
        toolbar.add_button(button4)
        toolbar.add_button(button5)
        toolbar.add_button(button6)
        return canvas


if __name__ == "__main__":
    # Save demo so that it doesn't get garbage collected when run within
    # existing event loop (i.e. from ipython).
    demo = demo_main(Demo, size=[700, 600])
