# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
from math import pi
import unittest

from numpy import array, allclose, ones, alltrue

from kiva import agg


class AffineMatrixTestCase(unittest.TestCase):
    def test_init(self):
        agg.AffineMatrix()

    def test_init_from_array(self):
        a = ones(6, "d")
        m = agg.AffineMatrix(a)
        desired = ones(6, "d")
        result = m.asarray()
        assert alltrue(result == desired)

    def test_init_from_array1(self):
        a = ones(6, "D")
        try:
            agg.AffineMatrix(a)
        except NotImplementedError:
            pass  # can't init from complex value.

    def test_init_from_array2(self):
        a = ones(7, "d")
        try:
            agg.AffineMatrix(a)
        except ValueError:
            pass  # can't init from array that isn't 6 element.

    def test_init_from_array3(self):
        a = ones((2, 3), "d")
        try:
            agg.AffineMatrix(a)
        except ValueError:
            pass  # can't init from array that isn't 1d.

    def test_imul(self):
        a = agg.AffineMatrix((2.0, 0, 0, 2.0, 0, 0))
        a *= a
        actual = a
        desired = agg.AffineMatrix((4.0, 0, 0, 4.0, 0, 0))
        assert alltrue(desired == actual)

    def test_asarray(self):
        m = agg.AffineMatrix()
        result = m.asarray()
        desired = array((1.0, 0.0, 0.0, 1.0, 0.0, 0.0))
        assert alltrue(result == desired)

    def _test_zero_arg_transform(self, method, orig, desired):
        m = agg.AffineMatrix(orig)
        method(m)
        result = m.asarray()
        assert alltrue(result == desired)

    def test_flip_x(self):
        method = agg.AffineMatrix.flip_x
        orig = array((1.0, 2.0, 3.0, 1.0, 4.0, 5.0))
        desired = array([-1.0, -2.0, 3.0, 1.0, -4.0, 5.0])
        self._test_zero_arg_transform(method, orig, desired)

    def test_flip_y(self):
        method = agg.AffineMatrix.flip_y
        orig = array((1.0, 2.0, 3.0, 1.0, 4.0, 5.0))
        desired = array([1.0, 2.0, -3.0, -1.0, 4.0, -5.0])
        self._test_zero_arg_transform(method, orig, desired)

    def test_reset(self):
        method = agg.AffineMatrix.reset
        orig = array((1.0, 2.0, 3.0, 1.0, 4.0, 5.0))
        desired = array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        self._test_zero_arg_transform(method, orig, desired)

    def test_multiply(self):
        orig = array((1.0, 2.0, 3.0, 1.0, 4.0, 5.0))
        desired = array([7.0, 4.0, 6.0, 7.0, 23.0, 18.0])
        m = agg.AffineMatrix(orig)
        other = agg.AffineMatrix(orig)
        m.multiply(other)
        result = m.asarray()
        assert alltrue(result == desired)

    def test_determinant(self):
        orig = array((1.0, 2.0, 3.0, 1.0, 4.0, 5.0))
        desired = -5.0
        m = agg.AffineMatrix(orig)
        result = m.determinant()
        assert alltrue(result == desired)

    def test_invert(self):
        orig = agg.AffineMatrix((1.0, 2.0, 3.0, 1.0, 4.0, 5.0))
        orig.invert()
        actual = orig.asarray()
        desired = array([-0.2, 0.4, 0.6, -0.2, -2.2, -0.6])
        assert allclose(desired, actual)

    def test_rotation_matrix(self):
        val = agg.rotation_matrix(pi / 2.0)
        desired = array([0.0, 1.0, -1.0, 0.0, 0.0, 0.0])
        actual = val.asarray()
        assert allclose(desired, actual)

    def test_translation_matrix(self):
        val = agg.translation_matrix(2.0, 3.0)
        desired = array([1.0, 0.0, 0.0, 1.0, 2.0, 3.0])
        actual = val.asarray()
        assert allclose(desired, actual)

    def test_scaling_matrix(self):
        val = agg.scaling_matrix(4.0, 4.0)
        desired = array([4.0, 0.0, 0.0, 4.0, 0.0, 0.0])
        actual = val.asarray()
        assert allclose(desired, actual)

    def test_skewing_matrix(self):
        val = agg.skewing_matrix(pi / 4.0, pi / 4.0)
        desired = array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0])
        actual = val.asarray()
        assert allclose(desired, actual)
