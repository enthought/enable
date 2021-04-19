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

from enable.compiled_path import CompiledPath
from enable.kiva_graphics_context import GraphicsContext
from enable.markers import CustomMarker


class TestCustomMarker(unittest.TestCase):

    def test_add_to_path(self):
        path = CompiledPath()
        path.begin_path()
        path.move_to(-5, -5)
        path.line_to(-5, 5)
        path.line_to(5, 5)
        path.line_to(5, -5)
        path.line_to(-5, -5)

        marker = CustomMarker(path=path)

        gc = GraphicsContext((50, 50))
        # should not fail
        marker.add_to_path(gc.get_empty_path(), np.int64(0))

    def test_get_compiled_path(self):
        path = CompiledPath()
        path.begin_path()
        path.move_to(-5, -5)
        path.line_to(-5, 5)
        path.line_to(5, 5)
        path.line_to(5, -5)
        path.line_to(-5, -5)

        marker = CustomMarker(path=path)

        # should not fail
        marker.get_compiled_path(np.int64(0))
