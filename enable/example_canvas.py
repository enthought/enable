# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
""" A suitable replacement for the old Canvas class in Kiva.
"""

from enable.api import Component, Window


class _DummyComponent(Component):
    def __init__(self, draw_func, *args, **kwargs):
        super(_DummyComponent, self).__init__(*args, **kwargs)
        self._draw_func = draw_func

    def __del__(self):
        self._draw_func = None

    def draw(self, gc, **kwargs):
        """ Call our wrapped draw function.
        """
        self._draw_func(gc)


class Canvas(Window):
    def __init__(self):
        # Create a component that wraps our do_draw method
        self.component = _DummyComponent(self.do_draw)

        # call our base class
        super(Window, self).__init__(None)

    def do_draw(self, gc):
        """ Method to be implemented by subclasses to actually perform various
        GC drawing commands before the GC is blitted into the screen.
        """
        pass
