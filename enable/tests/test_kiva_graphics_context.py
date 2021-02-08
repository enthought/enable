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

from enable.kiva_graphics_context import GraphicsContext


class TestGCErrors(unittest.TestCase):
    """Test some cases where a ValueError should be raised."""

    def test_bad_image_size(self):
        arr = np.array([[1, 2], [3, 4]], dtype=np.uint8)
        gc = GraphicsContext((50, 50))
        # The draw_image methods expects its first argument
        # to be a 3D whose last dimension has length 3 or 4.
        # Passing in arr should raise a value error.
        self.assertRaises(ValueError, gc.draw_image, arr)

        # Pass in a 3D array, but with an invalid size in the last dimension.
        self.assertRaises(ValueError, gc.draw_image, arr.reshape(2, 2, 1))
