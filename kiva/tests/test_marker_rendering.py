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
from kiva.marker_renderer import MarkerRenderer


class TestMarkerDrawing(unittest.TestCase):
    @contextlib.contextmanager
    def draw_and_check(self, buffer, check):
        # Start with a white backgroud
        buffer.fill(255)
        yield
        check(buffer)

    def exercise(self, renderer, buffer, check):
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
                with self.draw_and_check(buffer, check):
                    marker = getattr(constants, name)
                    retval = renderer.draw_markers(
                        points, 5, marker, fill, stroke
                    )
                    self.assertTrue(retval)

    def test_msb_alpha_32_bit(self):
        pixel_formats = ("abgr32", "argb32")

        def check(image):
            # Default is expected to be a totally white image.
            # Therefore we check if the whole image is white.
            if np.sum(image == [255, 255, 255, 255]) == (300 * 300 * 4):
                self.fail("The image looks empty, no pixels were drawn")

        buffer = np.empty((300, 300, 4), dtype=np.uint8)
        for pix_format in pixel_formats:
            gc = MarkerRenderer(buffer, pix_format=pix_format)
            self.exercise(gc, buffer, check)

    def test_lsb_alpha_32_bit(self):
        pixel_formats = ("bgra32", "rgba32")

        def check(image):
            # Default is expected to be a totally white image.
            # Therefore we check if the whole image is white.
            if np.sum(image == [255, 255, 255, 255]) == (300 * 300 * 4):
                self.fail("The image looks empty, no pixels were drawn")

        buffer = np.empty((300, 300, 4), dtype=np.uint8)
        for pix_format in pixel_formats:
            gc = MarkerRenderer(buffer, pix_format=pix_format)
            self.exercise(gc, buffer, check)

    def test_no_alpha_24_bit(self):
        pixel_formats = ("bgr24", "rgb24")

        def check(image):
            # Default is expected to be a totally white image.
            # Therefore we check if the whole image is white.
            if np.sum(image == [255, 255, 255]) == (300 * 300 * 3):
                self.fail("The image looks empty, no pixels were drawn")

        buffer = np.empty((300, 300, 3), dtype=np.uint8)
        for pix_format in pixel_formats:
            gc = MarkerRenderer(buffer, pix_format=pix_format)
            self.exercise(gc, buffer, check)

    def test_transformation(self):
        fill = (1.0, 0.0, 0.0, 1.0)
        stroke = (0.0, 0.0, 0.0, 1.0)
        buffer = np.empty((100, 100, 3), dtype=np.uint8)
        gc = MarkerRenderer(buffer, pix_format="rgb24")

        # Translate past the bounds
        gc.transform(1.0, 1.0, 0.0, 0.0, 110, 110)
        points = np.array([[0.0, 0.0]])
        buffer.fill(255)
        gc.draw_markers(points, 5, constants.SQUARE_MARKER, fill, stroke)
        # Transformed the point _out_ of the bounds. We expect nothing drawn
        all_white = (np.sum(buffer == [255, 255, 255]) == buffer.size)
        self.assertTrue(all_white)

        # Scale past the bounds
        gc.transform(2.0, 2.0, 0.0, 0.0, 0.0, 0.0)
        points = np.array([[90.0, 90.0]])
        gc.draw_markers(points, 5, constants.SQUARE_MARKER, fill, stroke)
        # Transformed the point _out_ of the bounds. We expect nothing drawn
        all_white = (np.sum(buffer == [255, 255, 255]) == buffer.size)
        self.assertTrue(all_white)

    def test_bad_arguments(self):
        fill = (1.0, 0.0, 0.0, 1.0)
        stroke = (0.0, 0.0, 0.0, 1.0)
        points = np.array([[1.0, 10.0], [50.0, 50.0], [42.0, 24.0]])
        buffer = np.empty((100, 100, 3), dtype=np.uint8)
        gc = MarkerRenderer(buffer, pix_format="rgb24")

        # Input array shape checking
        with self.assertRaises(ValueError):
            gc.draw_markers(fill, 5, constants.PLUS_MARKER, fill, stroke)
        with self.assertRaises(ValueError):
            gc.draw_markers(points, 5, constants.PLUS_MARKER, fill[:2], stroke)
        with self.assertRaises(ValueError):
            gc.draw_markers(points, 5, constants.PLUS_MARKER, fill, stroke[:2])

        # Argument type coercions
        with self.assertRaises(TypeError):
            gc.draw_markers(points, 5, "plus", fill, stroke)
        with self.assertRaises(TypeError):
            gc.draw_markers(points, [5], constants.PLUS_MARKER, fill, stroke)

        # Finally, check that drawing a bad marker ID returns False
        self.assertFalse(gc.draw_markers(points, 5, 500, fill, stroke))
