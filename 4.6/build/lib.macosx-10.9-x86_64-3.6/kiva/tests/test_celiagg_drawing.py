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

from kiva.celiagg import GraphicsContext
from kiva.tests.drawing_tester import DrawingImageTester


class TestCeliaggDrawing(DrawingImageTester, unittest.TestCase):
    def create_graphics_context(self, width, height, pixel_scale):
        return GraphicsContext((width, height), base_pixel_scale=pixel_scale)

    def test_clip_rect_transform(self):
        with self.draw_and_check():
            self.gc.clip_to_rect(0, 0, 100, 100)
            self.gc.begin_path()
            self.gc.rect(75, 75, 25, 25)
            self.gc.fill_path()
