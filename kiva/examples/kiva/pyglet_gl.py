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
from pyglet.window import key, Window

from kiva.api import STROKE

try:
    from kiva.gl import GraphicsContext
except ImportError as e:
    raise Exception(e)


class TestWindow(Window):
    """ Press Q or Escape to exit
    """

    def __init__(self, *args, **kw):
        Window.__init__(self, *args, **kw)
        self.init_window()

    def init_window(self):
        self.gc = GraphicsContext(size=(self.width, self.height))
        self.gc.gl_init()

    def on_key_press(self, symbol, modifiers):
        if symbol in (key.ESCAPE, key.Q):
            self.has_exit = True

    def draw(self):
        gc = self.gc
        with gc:
            gc.clear((0, 1, 0, 1))
            gc.set_stroke_color((1, 1, 1, 1))
            gc.set_line_width(2)
            pts = array([[50, 50], [50, 100], [100, 100], [100, 50]])
            gc.begin_path()
            gc.lines(pts)
            gc.close_path()
            gc.draw_path(STROKE)
            gc.flush()


def main():
    win = TestWindow(width=640, height=480)
    exit = False
    while not exit:
        win.switch_to()
        win.dispatch_events()
        win.clear()
        win.draw()
        win.flip()
        exit = win.has_exit


if __name__ == "__main__":
    main()
