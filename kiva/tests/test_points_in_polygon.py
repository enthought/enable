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
import warnings

from numpy import allclose, array, zeros

from kiva.agg import points_in_polygon as points_in_polygon_deprecated
from kiva.api import points_in_polygon


class TestPointsInPolygon(unittest.TestCase):
    def test_deprecated_import(self):
        polygon = array(((0.0, 0.0), (10.0, 0.0), (0.0, 10.0)))
        points = array(((5.0, 5.0),))

        with warnings.catch_warnings(record=True) as collector:
            warnings.simplefilter("always")
            points_in_polygon_deprecated(points, polygon)

            assert len(collector) == 1
            warn = collector[0]
            assert issubclass(warn.category, DeprecationWarning)

    def test_empty_points_in_polygon(self):
        polygon = array(((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)))
        points = zeros((0, 2))

        result = points_in_polygon(points, polygon)
        self.assertTrue(len(result) == len(points))

        polygon = array([])
        points = array(((-1.0, -1.0), (5.0, 5.0), (15.0, 15.0)))

        result = points_in_polygon(points, polygon)
        self.assertTrue(allclose(array([0, 0, 0]), result))

    def test_simple_points_in_polygon(self):
        polygon = array(((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)))
        points = array(((-1.0, -1.0), (5.0, 5.0), (15.0, 15.0)))

        result = points_in_polygon(points, polygon)
        self.assertTrue(allclose(array([0, 1, 0]), result))

    def test_transposed_points_in_polygon(self):
        polygon = array(((0.0, 10.0, 10.0, 0.0), (0.0, 0.0, 10.0, 10.0)))
        points = array(((-1.0, 5.0, 15.0), (-1.0, 5.0, 15.0)))

        result = points_in_polygon(points, polygon)
        self.assertTrue(allclose(array([0, 1, 0]), result))

    def test_asymmetric_points_in_polygon(self):
        polygon = array(((0.0, 0.0), (20.0, 0.0), (20.0, 10.0), (0.0, 10.0)))
        points = array(((5.0, 5.0), (10.0, 5.0), (15.0, 5.0)))

        result = points_in_polygon(points, polygon)
        self.assertTrue(allclose(array([1, 1, 1]), result))

    def test_rectangle(self):
        vertices = array(((0, 0), (0, 10), (10, 10), (10, 0)))

        # Try the lower left.
        trial = array(((0, 0),))
        oe_result = points_in_polygon(trial, vertices)
        w_result = points_in_polygon(trial, vertices, True)
        self.assertEqual(
            0, oe_result[0], "Lower left corner not in polygon. OEF"
        )
        self.assertEqual(
            1, w_result[0], "Lower left corner not in polygon. Winding"
        )

        # Try the center.
        trial = array(((5, 5),))
        oe_result = points_in_polygon(trial, vertices)
        w_result = points_in_polygon(trial, vertices, True)
        self.assertEqual(1, oe_result[0], "Center not in polygon. OEF")
        self.assertEqual(1, w_result[0], "Center not in polygon. Winding")

        # Try the center.
        trial = array(((10, 10),))
        oe_result = points_in_polygon(trial, vertices)
        w_result = points_in_polygon(trial, vertices, True)
        self.assertEqual(1, oe_result[0], "Top-right in polygon. OEF")
        self.assertEqual(0, w_result[0], "Top-right in polygon. Winding")

    def test_center_removed(self):
        # Tests a polygon which resembles the following:
        #
        # 9------8
        # |      |
        # |  3---+---------2
        # |  |   |         |
        # |  4---+----5    |
        # |      |    |    |
        # |      7----6    |
        # |                |
        # 0----------------1
        #
        # Using the winding rule, the inner square containing the edge (3,4)
        # is inside the polygon, while using the odd-even rule, it is outside.
        # The inner square with containing the edge (5,6) is outside in both
        # cases.

        vertices = array(
            (
                (0, 0),
                (10, 0),
                (10, 8),
                (2, 8),
                (2, 6),
                (8, 6),
                (8, 2),
                (5, 2),
                (5, 10),
                (0, 10),
            )
        )

        trial = array(((3, 7),))
        oe_result = points_in_polygon(trial, vertices)
        w_result = points_in_polygon(trial, vertices, True)
        self.assertEqual(0, oe_result[0], "Interior polygon inside: odd-even")
        self.assertEqual(1, w_result[0], "Interior polygon outside: winding")

        trial = array(((6, 5),))
        oe_result = points_in_polygon(trial, vertices)
        w_result = points_in_polygon(trial, vertices, True)
        self.assertEqual(0, oe_result[0], "Interior polygon inside: odd-even")
        self.assertEqual(0, w_result[0], "Interior polygon inside: winding")

    def test_discontiguous_inputs(self):
        polygon = array(((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)))
        points = array(((-1.0, -1.0), (5.0, 5.0), (15.0, 15.0), (7.0, 7.0)))

        # Create a discontiguous polygon array
        poly3 = zeros((polygon.shape[0], 3))
        poly3[:, :2] = polygon
        polygon = poly3[:, :2]

        # Create a discontiguous points array.
        points = points[1::2]

        result = points_in_polygon(points, polygon)
        self.assertTrue(allclose(array([1, 1]), result))
