# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
import unittest

import numpy as np

from kiva.tests.drawing_tester import DrawingImageTester
from kiva.image import GraphicsContext


class TestAggDrawing(DrawingImageTester, unittest.TestCase):
    def create_graphics_context(self, width, height, pixel_scale):
        return GraphicsContext((width, height), base_pixel_scale=pixel_scale)

    def test_save_dpi(self):
        # Base DPI is 72, but our default pixel scale is 2x.
        self.assertEqual(self.save_and_return_dpi(), 144)

    def test_unicode_gradient_args(self):
        color_nodes = [(0.0, 1.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)]
        with self.draw_and_check():
            w, h = self.gc.width(), self.gc.height()
            grad_stops = np.array(
                [(x, r, g, b, 1.0) for x, r, g, b in color_nodes]
            )

            self.gc.rect(0, 0, w, h)
            self.gc.linear_gradient(
                0, 0, w, 0, grad_stops, "pad", b"userSpaceOnUse"
            )
            self.gc.fill_path()

    def test_ipython_repr_png(self):
        self.gc.begin_path()
        self.gc.rect(75, 75, 25, 25)
        self.gc.fill_path()
        stream = self.gc._repr_png_()
        filename = "{0}.png".format(self.filename)
        with open(filename, 'wb') as fp:
            fp.write(stream)
        self.assertImageSavedWithContent(filename)
