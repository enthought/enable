# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
""" Small demo of the Label component.  """

from enable.example_support import DemoFrame, demo_main
from enable.label import Label


class Demo(DemoFrame):
    def _create_component(self):
        label = Label(bounds=[100, 50], position=[50, 50], text="HELLO")
        label.bgcolor = "lightpink"
        return label


if __name__ == "__main__":
    # Save demo so that it doesn't get garbage collected when run within
    # existing event loop (i.e. from ipython).
    demo = demo_main(Demo)
