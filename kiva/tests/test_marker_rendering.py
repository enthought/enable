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

import numpy as np

from kiva import constants
from kiva.marker_renderer import MarkerRendererRGB24


class TestMarkerDrawing(unittest.TestCase):
    def setUp(self):
        self.buffer = np.ones((300, 300, 3), dtype=np.uint8)
        self.gc = MarkerRendererRGB24(self.buffer)

    def tearDown(self):
        del self.gc

    @contextlib.contextmanager
    def draw_and_check(self):
        # Start with a white backgroud
        self.buffer.fill(255)
        yield
        self.assertImageContainsDrawing(self.buffer)

    def assertImageContainsDrawing(self, image):
        """ Check that there is something drawn.
        """
        # Default is expected to be a totally white image.
        # Therefore we check if the whole image is not white.
        if np.sum(image == [255, 255, 255]) != (300 * 300 * 3):
            return

        self.fail("The image looks empty, no pixels were drawn")

    def test_draw(self):
        marker_names = (
            "SQUARE_MARKER",
            "DIAMOND_MARKER",
            "CIRCLE_MARKER",
            "CROSSED_CIRCLE_MARKER",
            "CROSS_MARKER",
            "TRIANGLE_MARKER",
            "INVERTED_TRIANGLE_MARKER",
            "PLUS_MARKER",
            "DOT_MARKER",
            "PIXEL_MARKER",
        )
        fill = (1.0, 0.0, 0.0, 1.0)
        stroke = (0.0, 0.0, 0.0, 1.0)
        count = 1000
        points = (np.random.random(size=count) * 300.0)
        points = points.reshape(count // 2, 2)

        for name in marker_names:
            with self.subTest(msg=name):
                with self.draw_and_check():
                    marker = getattr(constants, name)
                    self.gc.draw_marker_at_points(
                        points, 5, marker, fill, stroke
                    )
