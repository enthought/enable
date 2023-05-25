# (C) Copyright 2005-2023 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!

import unittest

from enable.drawing.point_line import PointLine


class TestPointLine(unittest.TestCase):

    def test_pointer_shapes(self):
        point_line = PointLine()

        self.assertEqual(point_line.normal_cursor, "arrow")
        self.assertEqual(point_line.drawing_cursor, "pencil")
        self.assertEqual(point_line.delete_cursor, "bullseye")
        self.assertEqual(point_line.move_cursor, "sizing")
