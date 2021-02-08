# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
from enable.api import Container, Window, TextField
from enable.example_support import DemoFrame, demo_main
from enable.stacked_container import VStackedContainer, HStackedContainer

size = (240, 240)


class Demo(DemoFrame):
    def _create_window(self):
        return Window(self, -1, component=self._create_component())

    def _create_component(self):
        stack = VStackedContainer(
            position=[0, 0],
            halign="center",
            valign="center",  # border_visible=True,
            fit_components="hv",
            auto_size=True,
            stack_order="top_to_bottom",
            bgcolor="red",
        )

        strings = [
            "apple",
            "banana",
            "cherry",
            "durian",
            "eggfruit",
            "fig",
            "grape",
            "honeydew",
        ]

        for i, s in enumerate(strings):
            label = TextField(
                text=s,
                resizable="",
                bounds=[100 + i * 10, 20],
                bgcolor="red",  # border_visible=True,
                text_offset=1,
            )
            number = TextField(
                text=str(i + 1),
                resizable="",
                bgcolor="blue",  # border_visible=True,
                text_offset=1,
                can_edit=False,
                bounds=[20, 20],
            )
            row = HStackedContainer(
                fit_components="hv",
                auto_size=True,
                resizable="",
                valign="top",
                border_visible=True,
            )
            row.add(number)
            row.add(label)
            stack.add(row)

        container = Container(position=[20, 20], bounds=size)
        container.add(stack)
        container2 = Container(bounds=size)
        container2.add(container)
        return container2


if __name__ == "__main__":
    # Save demo so that it doesn't get garbage collected when run within
    # existing event loop (i.e. from ipython).
    demo = demo_main(Demo, size=size)
