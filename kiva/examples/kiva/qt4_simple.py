# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
import sys

from numpy import array

# This is PyQt specific so force the toolkit selection.
from traits.etsconfig.api import ETSConfig

ETSConfig.toolkit = "qt4.qpainter"

from enable.example_canvas import Canvas

try:
    from pyface.qt import QtGui
except ImportError:
    raise Exception("PyQt4 needs to be installed to run this example")


class MyCanvas(Canvas):
    def do_draw(self, gc):
        w = gc.width()
        h = gc.height()

        # Draw a red gradient filled box with green border
        gc.rect(w / 4, h / 4, w / 2, h / 2)
        gc.set_line_width(5.0)
        gc.set_stroke_color((0.0, 1.0, 0.0, 1.0))

        start = array([0.0, 1.0, 0.0, 0.0, 1.0])
        end = array([1.0, 1.0, 1.0, 1.0, 1.0])
        gc.radial_gradient(
            w / 4, h / 4, 200, w / 4 + 100, h / 4 + 100,
            array([start, end]),
            "reflect",
        )
        gc.draw_path()

        # draw a black rect with rounded corners
        r = max(1, min(w, h) / 10)
        gc.set_line_width(2.0)
        gc.set_stroke_color((0.0, 0.0, 0.0, 1.0))
        gc.move_to(w / 3, h / 2)

        gc.arc_to(w / 3, 2 * h / 3,
                  w / 2, 2 * h / 3,
                  r)
        gc.arc_to(2 * w / 3, 2 * h / 3,
                  2 * w / 3, h / 2,
                  r)
        gc.arc_to(2 * w / 3, h / 3,
                  w / 2, h / 3,
                  r)
        gc.arc_to(w / 3, h / 3,
                  w / 3, h / 2,
                  r)
        gc.line_to(w / 3, h / 2)
        gc.stroke_path()


if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)

    w = MyCanvas()
    w.control.resize(500, 500)
    w.control.setWindowTitle("Simple Kiva.qt4 example")
    w.control.show()

    app.exec_()
