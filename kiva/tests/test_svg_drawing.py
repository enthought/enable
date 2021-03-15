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
from xml.etree import ElementTree
import unittest

from kiva.svg import GraphicsContext
from kiva.tests.drawing_tester import DrawingTester


class TestSVGDrawing(DrawingTester, unittest.TestCase):
    def create_graphics_context(self, width, height):
        return GraphicsContext((width, height))

    @contextlib.contextmanager
    def draw_and_check(self):
        yield
        filename = "{0}.svg".format(self.filename)
        self.gc.save(filename)
        tree = ElementTree.parse(filename)
        elements = [element for element in tree.iter()]
        if not len(elements) in [4, 7]:
            self.fail("The expected number of elements was not found")

    def test_ipython_repr_svg(self):
        self.gc.begin_path()
        self.gc.rect(75, 75, 25, 25)
        self.gc.fill_path()
        stream = self.gc._repr_svg_()
        filename = "{0}.svg".format(self.filename)
        with open(filename, 'w', encoding='utf8') as fp:
            fp.write(stream)
        tree = ElementTree.parse(filename)
        elements = [element for element in tree.iter()]
        if not len(elements) in [4, 7]:
            self.fail("The expected number of elements was not found")
