# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
from numpy import array

from enable.api import Container
from enable.example_support import DemoFrame, demo_main
from enable.text_grid import TextGrid

size = (400, 100)


class Demo(DemoFrame):
    def _create_component(self):
        strings = array(
            [
                ["apple", "banana", "cherry", "durian"],
                ["eggfruit", "fig", "grape", "honeydew"],
            ]
        )
        grid = TextGrid(string_array=strings)
        container = Container(bounds=size)
        container.add(grid)
        return container


if __name__ == "__main__":
    # Save demo so that it doesn't get garbage collected when run within
    # existing event loop (i.e. from ipython).
    demo = demo_main(Demo, size=size)
