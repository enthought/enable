from math import pi
import unittest

from numpy import array, allclose, ones, alltrue

from kiva import agg

class AffineMatrixTestCase(unittest.TestCase):

    def test_init(self):
        m = agg.AffineMatrix()

    def test_init_from_array(self):
        a = ones(6,'d')
        m = agg.AffineMatrix(a)
        desired = ones(6,'d')
        result = m.asarray()
        assert(alltrue(result == desired))

    def test_init_from_array1(self):
        a = ones(6,'D')
        try:
            m = agg.AffineMatrix(a)
        except NotImplementedError:
            pass # can't init from complex value.

    def test_init_from_array2(self):
        a = ones(7,'d')
        try:
            m = agg.AffineMatrix(a)
        except ValueError:
            pass # can't init from array that isn't 6 element.

    def test_init_from_array3(self):
        a = ones((2,3),'d')
        try:
            m = agg.AffineMatrix(a)
        except ValueError:
            pass # can't init from array that isn't 1d.

    def test_imul(self):
        a = agg.AffineMatrix((2.0,0,0,2.0,0,0))
        a *= a
        actual = a
        desired = agg.AffineMatrix((4.0,0,0,4.0,0,0))
        assert(alltrue(desired==actual))

    def test_asarray(self):
        m = agg.AffineMatrix()
        result = m.asarray()
        desired = array((1.0,0.0,0.0,1.0,0.0,0.0))
        assert(alltrue(result == desired))

    def _test_zero_arg_transform(self,method, orig, desired):
        m = agg.AffineMatrix(orig)
        method(m)
        result = m.asarray()
        assert(alltrue(result == desired))

    def test_flip_x(self):
        method = agg.AffineMatrix.flip_x
        orig = array((1.0,2.0,3.0,1.0,4.0,5.0))
        desired = array([-1.,-2.,3.,1.,-4.,5.])
        self._test_zero_arg_transform(method,orig,desired)

    def test_flip_y(self):
        method = agg.AffineMatrix.flip_y
        orig = array((1.0,2.0,3.0,1.0,4.0,5.0))
        desired = array([ 1.,2.,-3.,-1.,4.,-5.])
        self._test_zero_arg_transform(method,orig,desired)

    def test_reset(self):
        method = agg.AffineMatrix.reset
        orig = array((1.0,2.0,3.0,1.0,4.0,5.0))
        desired = array([ 1.,0.,0.,1.,0.,0.])
        self._test_zero_arg_transform(method,orig,desired)

    def test_multiply(self):
        orig = array((1.0,2.0,3.0,1.0,4.0,5.0))
        desired = array([  7.,4.,6.,7.,23.,18.])
        m = agg.AffineMatrix(orig)
        other = agg.AffineMatrix(orig)
        m.multiply(other)
        result = m.asarray()
        assert(alltrue(result == desired))

    def test_determinant(self):
        orig = array((1.0,2.0,3.0,1.0,4.0,5.0))
        desired = -0.2
        m = agg.AffineMatrix(orig)
        result = m.determinant()
        assert(alltrue(result == desired))

    def test_invert(self):
        orig = agg.AffineMatrix((1.0,2.0,3.0,1.0,4.0,5.0))
        orig.invert()
        actual = orig.asarray()
        desired = array([-0.2,0.4,0.6,-0.2,-2.2,-0.6])
        assert(allclose(desired,actual))

    def test_rotation_matrix(self):
        val = agg.rotation_matrix(pi/2.)
        desired = array([ 0.0,1.0,-1.0,0.0,0.0,0.0])
        actual = val.asarray()
        assert(allclose(desired,actual))

    def test_translation_matrix(self):
        val = agg.translation_matrix(2.0,3.0)
        desired = array([ 1.0,0.0,0.0,1.0,2.0,3.0])
        actual = val.asarray()
        assert(allclose(desired,actual))

    def test_scaling_matrix(self):
        val = agg.scaling_matrix(4.0,4.0)
        desired = array([ 4.0,0.0,0.0,4.0,0.0,0.0])
        actual = val.asarray()
        assert(allclose(desired,actual))

    def test_skewing_matrix(self):
        val = agg.skewing_matrix(pi/4.,pi/4.)
        desired = array([ 1.0,1.0,1.0,1.0,0.0,0.0])
        actual = val.asarray()
        assert(allclose(desired,actual))

#----------------------------------------------------------------------------
# test setup code.
#----------------------------------------------------------------------------

def test_suite(level=1):
    suites = []
    if level > 0:
        suites.append( unittest.makeSuite(AffineMatrixTestCase,'test_') )
    total_suite = unittest.TestSuite(suites)
    return total_suite

def test(level=10):
    all_tests = test_suite(level)
    runner = unittest.TextTestRunner()
    runner.run(all_tests)
    return runner

if __name__ == "__main__":
    test()
