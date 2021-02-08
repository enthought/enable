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
**WARNING**

  This demo might not work as expected and some documented features might be
  missing.

"""
# Issue related to the demo warning: enthought/enable#502

import os.path
import xml.etree.cElementTree as etree

from enable.api import BaseTool, Component, ComponentEditor, Container
from enable.savage.svg.backends.kiva.renderer import Renderer as KivaRenderer
from enable.savage.svg.document import SVGDocument
from kiva.api import MODERN, Font
from traits.api import Callable, Enum, HasTraits, Instance, List, Str
from traitsui.api import Item, View


class CanvasButton(Component):
    document = Instance(SVGDocument)
    toggle_document = Instance(SVGDocument)
    label = Str()
    callback = Callable
    callback_args = List(Str)
    state = Enum("up", "down")

    bounds = [64, 64]

    def __init__(self, filename, callback, callback_args, *args, **kw):
        super(CanvasButton, self).__init__(*args, **kw)

        self.document = self._load_svg_document(filename)

        # set the toggle doc if it wasn't passed in as a keyword arg
        if self.toggle_document is None:
            toggle_filename = os.path.join(
                os.path.dirname(__file__), "button_toggle.svg"
            )
            self.toggle_document = self._load_svg_document(toggle_filename)

        self.callback = callback
        self.callback_args = callback_args

    def draw(self, gc, view_bounds, mode):
        if self.state == "down":
            self._draw_svg_document(gc, self.toggle_document)

        self._draw_svg_document(gc, self.document)

        if len(self.label) > 0:
            self._draw_label(gc)

    def _load_svg_document(self, filename):
        if not os.path.exists(filename):
            raise ValueError
        tree = etree.parse(filename)
        root = tree.getroot()
        return SVGDocument(root, renderer=KivaRenderer)

    def _draw_svg_document(self, gc, document):
        with gc:
            gc.translate_ctm(self.x, self.y + self.height)
            doc_size = document.getSize()
            gc.scale_ctm(
                self.width / float(doc_size[0]),
                -self.height / float(doc_size[1]),
            )
            document.render(gc)

    def _draw_label(self, gc):
        with gc:
            font = Font(family=MODERN)
            gc.set_font(font)

            _x, _y, width, height = gc.get_text_extent(self.label)
            text_x = self.x + (self.width - width) / 2.0
            text_y = self.y - height

            gc.show_text(self.label, (text_x, text_y))

    def perform(self):
        self.callback(*self.callback_args)


class ButtonCanvas(Container):
    def draw(self, gc, view_bounds=None, mode="default"):
        for component in self.components:
            component.draw(gc, view_bounds, mode)

    def add_button(self, button):
        button.container = self
        self.components.append(button)


class ButtonSelectionTool(BaseTool):
    """ Listens for double-clicks and tries to open a traits editor on the
        graph node under the mouse.
    """

    def normal_left_down(self, event):
        for component in self.component.components:
            if (isinstance(component, CanvasButton)
                    and component.is_in(event.x, event.y)):
                component.state = "down"
                component.request_redraw()
                component.perform()
                break

    def normal_left_up(self, event):
        for component in self.component.components:
            if isinstance(component, CanvasButton):
                component.state = "up"
                component.request_redraw()


class ButtonCanvasView(HasTraits):
    canvas = Instance(Container)

    traits_view = View(
        Item("canvas", editor=ComponentEditor(), show_label=False),
        width=400,
        height=400,
        resizable=True,
    )

    def __init__(self, *args, **kw):
        super(ButtonCanvasView, self).__init__(*args, **kw)
        self.add_buttons()

    def _canvas_default(self):
        """ default setter for _canvas
        """
        container = ButtonCanvas()
        container.tools.append(ButtonSelectionTool(component=container))
        return container

    def add_buttons(self):
        data_dir = os.path.dirname(__file__)
        self.canvas.add_button(
            CanvasButton(
                os.path.join(data_dir, "edit-copy.svg"),
                self.do_copy,
                [],
                label="Copy",
                x=150,
                y=150,
            )
        )
        self.canvas.add_button(
            CanvasButton(
                os.path.join(data_dir, "edit-paste.svg"),
                self.do_paste,
                [],
                label="Paste",
                x=250,
                y=150,
            )
        )

    def do_copy(self):
        print("copying something")

    def do_paste(self):
        print("pasting something")


demo = ButtonCanvasView()

if __name__ == "__main__":
    demo.configure_traits()
