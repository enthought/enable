# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
""" Test suite for affine transforms.

    :Author:      Eric Jones, Enthought, Inc., eric@enthought.com
    :Copyright:   Space Telescope Science Institute
    :License:     BSD Style

    So far, this is mainly a "smoke test" suite to make sure
    nothing is obviously wrong.  It relies on the transforms
    being stored in 3x3 array.
"""
import unittest

from numpy import allclose, alltrue, array, cos, dot, identity, pi, ravel

from kiva import affine


class AffineConstructorsTestCase(unittest.TestCase):
    def test_identity(self):
        i = affine.affine_identity()
        self.assertTrue(allclose(identity(3), i))

    def test_from_values(self):
        a, b, c, d, tx, ty = 1, 2, 3, 4, 5, 6
        mat = affine.affine_from_values(a, b, c, d, tx, ty)
        desired = array([[a, b, 0], [c, d, 0], [tx, ty, 1]])
        assert alltrue(ravel(mat) == ravel(desired))

    def test_from_scale(self):
        transform = affine.affine_from_scale(5.0, 6.0)
        pt1 = array([1.0, 1.0, 1.0])
        actual = dot(pt1, transform)
        desired = pt1 * array((5.0, 6.0, 1.0))
        assert alltrue(actual == desired)

    def test_from_translation(self):
        transform = affine.affine_from_translation(5.0, 6.0)
        pt1 = array([1.0, 1.0, 1.0])
        actual = dot(pt1, transform)
        desired = pt1 + array((5.0, 6.0, 0.0))
        assert alltrue(actual == desired)

    def test_from_rotation(self):
        transform = affine.affine_from_rotation(pi / 4.0)
        pt1 = array([1.0, 0.0, 1.0])
        actual = dot(pt1, transform)
        cos_pi_4 = cos(pi / 4.0)
        desired = array((cos_pi_4, cos_pi_4, 1.0))
        assert alltrue((actual - desired) < 1e-6)


class AffineOperationsTestCase(unittest.TestCase):
    """ Test are generally run by operating on a matrix and using
        it to transform a point.  We then transform the point using
        some known sequence of operations that should produce the
        same results.
    """

    def test_scale(self):
        a, b, c, d, tx, ty = 1, 2, 3, 4, 5, 6
        transform1 = affine.affine_from_values(a, b, c, d, tx, ty)
        transform2 = affine.scale(transform1, 0.5, 1.5)
        pt1 = array([1.0, -1.0, 1.0])
        actual = dot(pt1, transform2)
        # this does the first transform and the scaling separately
        desired = dot(pt1, transform1) * array((0.5, 1.5, 1.0))
        assert alltrue((actual - desired) < 1e-6)

    def test_translate(self):
        a, b, c, d, tx, ty = 1, 2, 3, 4, 5, 6
        transform1 = affine.affine_from_values(a, b, c, d, tx, ty)
        translate_transform = array([[0, 0, 0], [0, 0, 0], [0.5, 1.5, 1]])
        tot_transform = affine.translate(transform1, 0.5, 1.5)
        pt1 = array([1.0, -1.0, 1.0])
        actual = dot(pt1, tot_transform)
        # this does the first transform and the translate separately
        desired = dot(dot(pt1, translate_transform), transform1)
        assert alltrue((actual - desired) < 1e-6)

    def test_rotate(self):
        a, b, c, d, tx, ty = 1.0, 0, 0, 1.0, 0, 0
        transform1 = affine.affine_from_values(a, b, c, d, tx, ty)
        tot_transform = affine.rotate(transform1, pi / 4)
        pt1 = array([1.0, 0.0, 1.0])
        actual = dot(pt1, tot_transform)
        # this does the first transform and the translate separately
        cos_pi_4 = 0.707_106_781_186_547_57
        desired = array((cos_pi_4, cos_pi_4, 1.0))
        assert alltrue((actual - desired) < 1e-6)

    def test_invert(self):
        """ An matrix times its inverse should produce the identity matrix
        """
        a, b, c, d, tx, ty = 1, 2, 3, 4, 5, 6
        transform1 = affine.affine_from_values(a, b, c, d, tx, ty)
        transform2 = affine.invert(transform1)
        desired = affine.affine_identity()
        actual = dot(transform2, transform1)
        assert alltrue((ravel(actual) - ravel(desired)) < 1e-6)

    def test_concat(self):
        a, b, c, d, tx, ty = 1, 2, 3, 4, 5, 6
        transform1 = affine.affine_from_values(a, b, c, d, tx, ty)
        a, b, c, d, tx, ty = 2, 3, 4, 5, 6, 7
        transform2 = affine.affine_from_values(a, b, c, d, tx, ty)
        tot_transform = affine.concat(transform1, transform2)
        pt1 = array([1.0, -1.0, 1.0])
        actual = dot(pt1, tot_transform)
        # this does the first transform and the scaling separately
        desired = dot(dot(pt1, transform2), transform1)
        assert alltrue((actual - desired) < 1e-6)


class AffineInformationTestCase(unittest.TestCase):
    """ Test are generally run by operating on a matrix and using
        it to transform a point.  We then transform the point using
        some known sequence of operations that should produce the
        same results.
    """

    def test_is_identity(self):
        # a true case.
        m = affine.affine_identity()
        assert affine.is_identity(m)
        # and a false one.
        a, b, c, d, tx, ty = 1, 2, 3, 4, 5, 6
        m = affine.affine_from_values(a, b, c, d, tx, ty)
        assert not affine.is_identity(m)

    def test_affine_params(self):
        a, b, c, d, tx, ty = 1, 2, 3, 4, 5, 6
        trans = affine.affine_from_values(a, b, c, d, tx, ty)
        aa, bb, cc, dd, txx, tyy = affine.affine_params(trans)
        assert (a, b, c, d, tx, ty) == (aa, bb, cc, dd, txx, tyy)

    def test_trs_factor(self):
        trans = affine.affine_identity()
        trans = affine.translate(trans, 6, 5)
        trans = affine.rotate(trans, 2.4)
        trans = affine.scale(trans, 0.2, 10)
        tx, ty, sx, sy, angle = affine.trs_factor(trans)
        assert (tx, ty) == (6, 5)
        assert (sx, sy) == (0.2, 10)
        assert angle == 2.4

    def test_tsr_factor(self):
        trans = affine.affine_identity()
        trans = affine.translate(trans, 6, 5)
        trans = affine.scale(trans, 0.2, 10)
        trans = affine.rotate(trans, 2.4)
        tx, ty, sx, sy, angle = affine.tsr_factor(trans)
        assert (tx, ty) == (6, 5)
        assert (sx, sy) == (0.2, 10)
        assert angle == 2.4


class TransformPointsTestCase(unittest.TestCase):
    def test_transform_point(self):
        pt = array((1, 1))
        ctm = affine.affine_identity()
        ctm = affine.translate(ctm, 5, 5)
        new_pt = affine.transform_point(ctm, pt)
        assert alltrue(new_pt == array((6, 6)))

        ctm = affine.rotate(ctm, pi)
        new_pt = affine.transform_point(ctm, pt)
        assert sum(new_pt - array((4.0, 4.0))) < 1e-15

        ctm = affine.scale(ctm, 10, 10)
        new_pt = affine.transform_point(ctm, pt)
        assert sum(new_pt - array((-5.0, -5.0))) < 1e-15

    def test_transform_points(self):
        # not that thorough...
        pt = array(((1, 1),))
        ctm = affine.affine_identity()
        ctm = affine.translate(ctm, 5, 5)
        new_pt = affine.transform_points(ctm, pt)
        assert alltrue(new_pt[0] == array((6, 6)))

        ctm = affine.rotate(ctm, pi)
        new_pt = affine.transform_points(ctm, pt)
        assert sum(new_pt[0] - array((4.0, 4.0))) < 1e-15

        ctm = affine.scale(ctm, 10, 10)
        new_pt = affine.transform_points(ctm, pt)
        assert sum(new_pt[0] - array((-5.0, -5.0))) < 1e-15
