from enthought.util.numerix import *
import unittest


from enthought.kiva import agg

from test_utils import *

class RgbaTestCase(unittest.TestCase):

    def test_init(self):
        m = agg.Rgba()

    def test_init1(self):
        m = agg.Rgba(.5,.5,.5)
        desired = array((.5,.5,.5,1.0))
        assert_arrays_equal(m.asarray(),desired)

    def test_init_from_array1(self):
        a = ones(3,'d') * .8
        m = agg.Rgba(a)
        desired = ones(4,'d') * .8
        desired[3] = 1.0
        result = m.asarray()
        assert_arrays_equal(result, desired)

    def test_init_from_array2(self):
        a = ones(4,'d') * .8
        m = agg.Rgba(a)
        desired = ones(4,'d') * .8
        result = m.asarray()
        assert_arrays_equal(result, desired)

    def test_init_from_array3(self):
        a = ones(5,'d')
        try:
            m = agg.Rgba(a)
        except ValueError:
            pass # can't init from array that isn't 6 element.

    def test_init_from_array4(self):
        a = ones((2,3),'d')
        try:
            m = agg.Rgba(a)
        except ValueError:
            pass # can't init from array that isn't 1d.

    def test_gradient(self):
        first = agg.Rgba(0.,0.,0.,0.)
        second = agg.Rgba(1.,1.,1.,1.)
        actual = first.gradient(second,.5).asarray()
        desired = array((.5,.5,.5,.5))        
        assert_arrays_equal(actual, desired)
        
    def test_pre(self):
        first = agg.Rgba(1.0,1.0,1.0,0.5)
        actual = first.premultiply().asarray()
        desired = array((.5,.5,.5,.5))        
        assert_arrays_equal(actual, desired)

if __name__ == "__main__":
    unittest.main()
