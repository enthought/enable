# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
from enable.api import Container, Label, NativeScrollBar
from enable.example_support import DemoFrame, demo_main


class Demo(DemoFrame):
    def _create_component(self):
        label = Label(
            text="h:\nv:",
            font="modern 16",
            position=[20, 50],
            bounds=[100, 100],
            bgcolor="red",
            color="white",
            hjustify="center",
            vjustify="center",
        )

        vscroll = NativeScrollBar(
            orientation="vertical",
            bounds=[15, label.height],
            position=[label.x2, label.y],
            range=(0, 100.0, 10.0, 1.0),
            enabled=True,
        )
        vscroll.on_trait_change(self._update_vscroll, "scroll_position")

        hscroll = NativeScrollBar(
            orientation="horizontal",
            bounds=[label.width, 15],
            position=[label.x, label.y - 15],
            range=(0, 100.0, 10.0, 1.0),
            enabled=True,
        )
        hscroll.on_trait_change(self._update_hscroll, "scroll_position")

        container = Container(
            bounds=[200, 200], border_visible=True, padding=15
        )
        container.add(label, hscroll, vscroll)
        container.on_trait_change(self._update_layout, "bounds")
        container.on_trait_change(self._update_layout, "bounds_items")

        self.label = label
        self.hscroll = hscroll
        self.vscroll = vscroll
        return container

    def _update_hscroll(self):
        text = self.label.text.split("\n")
        text[0] = "h: " + str(self.hscroll.scroll_position)
        self.label.text = "\n".join(text)

    def _update_vscroll(self):
        text = self.label.text.split("\n")
        text[1] = "v: " + str(self.vscroll.scroll_position)
        self.label.text = "\n".join(text)

    def _update_layout(self):
        self.vscroll._widget_moved = True
        self.hscroll._widget_moved = True


if __name__ == "__main__":
    # Save demo so that it doesn't get garbage collected when run within
    # existing event loop (i.e. from ipython).
    demo = demo_main(Demo, title="Scrollbar demo", size=(250, 250))
