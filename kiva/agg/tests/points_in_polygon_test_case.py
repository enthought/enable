import unittest

from numpy import array, allclose

from kiva import agg

class TestPointsInPolygon(unittest.TestCase):

    def test_simple_points_in_polygon(self):

        polygon = array(((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)))
        points = array(((-1.0, -1.0), (5.0, 5.0), (15.0, 15.0)))

        result = agg.points_in_polygon(points, polygon)
        self.assertTrue(allclose(array([0,1,0]), result))

        return

    def test_asymmetric_points_in_polygon(self):

        polygon = array(((0.0, 0.0), (20.0, 0.0), (20.0, 10.0), (0.0, 10.0)))
        points = array(((5.0, 5.0), (10.0, 5.0), (15.0, 5.0)))

        result = agg.points_in_polygon(points, polygon)
        self.assertTrue(allclose(array([1,1,1]), result))

        return


    def test_rectangle(self):

        vertices = array(((0,0), (0,10), (10,10), (10,0)))

        # Try the lower left.
        trial = array(((0,0),))
        oe_result = agg.points_in_polygon(trial, vertices)
        w_result = agg.points_in_polygon(trial, vertices, True)
        self.assertEqual(0, oe_result[0],
                         "Lower left corner not in polygon. OEF")
        self.assertEqual(1, w_result[0],
                         "Lower left corner not in polygon. Winding")

        # Try the center.
        trial = array(((5,5),))
        oe_result = agg.points_in_polygon(trial, vertices)
        w_result = agg.points_in_polygon(trial, vertices, True)
        self.assertEqual(1, oe_result[0],
                         "Center not in polygon. OEF")
        self.assertEqual(1, w_result[0],
                         "Center not in polygon. Winding")

        # Try the center.
        trial = array(((10,10),))
        oe_result = agg.points_in_polygon(trial, vertices)
        w_result = agg.points_in_polygon(trial, vertices, True)
        self.assertEqual(1, oe_result[0],
                         "Top-right in polygon. OEF")
        self.assertEqual(0, w_result[0],
                         "Top-right in polygon. Winding")

        return

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

        vertices = array(((0,0),
                          (10, 0),
                          (10, 8),
                          (2, 8),
                          (2, 6),
                          (8, 6),
                          (8, 2),
                          (5, 2),
                          (5, 10),
                          (0, 10)))

        trial = array(((3,7),))
        oe_result = agg.points_in_polygon(trial, vertices)
        w_result = agg.points_in_polygon(trial, vertices, True)
        self.assertEqual(0, oe_result[0],
                         "Interior polygon inside: odd-even")
        self.assertEqual(1, w_result[0],
                         "Interior polygon outside: winding")

        trial = array(((6,5),))
        oe_result = agg.points_in_polygon(trial, vertices)
        w_result = agg.points_in_polygon(trial, vertices, True)
        self.assertEqual(0, oe_result[0],
                         "Interior polygon inside: odd-even")
        self.assertEqual(0, w_result[0],
                         "Interior polygon inside: winding")


if __name__ == "__main__":
          unittest.main()

#### EOF ######################################################################
