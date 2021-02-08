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
This demonstrates the use of the drop tool.
"""

from enable.example_support import DemoFrame, demo_main
from enable.api import Component, Container, Label
from enable.tools.base_drop_tool import BaseDropTool


class Box(Component):

    resizable = ""

    def _draw_mainlayer(self, gc, view_bounds=None, mode="default"):
        with gc:
            gc.set_fill_color((1.0, 0.0, 0.0, 1.0))
            dx, dy = self.bounds
            x, y = self.position
            gc.rect(x, y, dx, dy)
            gc.fill_path()


class TextDropTool(BaseDropTool):
    """ Example implementation of a drop tool """

    def accept_drop(self, location, obj):
        return True

    def handle_drop(self, location, objs):
        if not isinstance(objs, list):
            objs = [objs]
        x, y = location
        for obj in objs:
            label = Label(text=str(obj), position=[x, y], bounds=[100, 50])
            self.component.add(label)
            self.component.request_redraw()
            y += 15


class MyFrame(DemoFrame):
    def _create_component(self):
        box = Box(bounds=[100.0, 100.0], position=[50.0, 50.0])
        container = Container(bounds=[500, 500])
        container.add(box)
        drop_tool = TextDropTool(component=container)
        container.tools.append(drop_tool)

        return container


if __name__ == "__main__":
    demo_main(MyFrame)
