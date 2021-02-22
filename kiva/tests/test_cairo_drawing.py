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

from kiva.tests.drawing_tester import DrawingImageTester

try:
    import cairo  # noqa
except ImportError:
    CAIRO_NOT_AVAILABLE = True
else:
    CAIRO_NOT_AVAILABLE = False


@unittest.skipIf(CAIRO_NOT_AVAILABLE, "Cannot import cairo")
class TestCairoDrawing(DrawingImageTester, unittest.TestCase):
    def create_graphics_context(self, width, height, pixel_scale):
        from kiva.cairo import GraphicsContext

        return GraphicsContext((width, height), base_pixel_scale=pixel_scale)
