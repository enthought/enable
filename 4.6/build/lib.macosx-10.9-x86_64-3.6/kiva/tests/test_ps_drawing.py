# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
import contextlib
import unittest

from kiva.ps import PSGC
from kiva.tests.drawing_tester import DrawingTester


class TestPSDrawing(DrawingTester, unittest.TestCase):
    def create_graphics_context(self, width, height):
        return PSGC((width, height))

    @contextlib.contextmanager
    def draw_and_check(self):
        yield
        filename = "{0}.eps".format(self.filename)
        self.gc.save(filename)
        with open(filename, "r") as handle:
            lines = handle.readlines()

        # Just a simple check that the path has been closed or the text has
        # been drawn.
        line = lines[-1].strip()
        if not any((line.endswith("fill"),
                    line.endswith("stroke"),
                    line.endswith("cliprestore"),
                    line.endswith("grestore"),
                    "(hello kiva) show\n" in lines)):
            self.fail("Path was not closed")
