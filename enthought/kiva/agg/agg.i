/* -*- c++ -*- */
/* File : example.i */
%module agg

#if (SWIG_VERSION > 0x010322)
%feature("compactdefaultargs");
#endif // (SWIG_VERSION > 0x010322)

%include "constants.i"
%include "rgba.i"
%include "font_type.i"
%include "affine_matrix.i"
%include "compiled_path.i"
//%include "image.i"
%include "graphics_context.i"
%include "hit_test.i"

%pythoncode {
#---- testing ----#
import sys
import os

def test(level=10):
    import unittest
    runner = unittest.TextTestRunner()
    runner.run(test_suite())
    return runner

def test_suite(level=1):
    # a bit of a mess right now
    import scipy_test.testing
    import agg
    agg_path = os.path.dirname(os.path.abspath(agg.__file__))
    sys.path.insert(0,agg_path)
    sys.path.insert(0,os.path.join(agg_path,'tests'))

    suites = []
    import test_affine_matrix
    suites.append( test_affine_matrix.test_suite(level=level) )

    """
    import test_font_type
    suites.append( test_font_type.test_suite(level=level) )
    
    import test_rgba
    suites.append( test_rgba.test_suite(level=level) )
    """

    import test_compiled_path
    suites.append( test_compiled_path.test_suite(level=level) )

    import test_graphics_context
    suites.append( test_graphics_context.test_suite(level=level) )

    """
    import test_image
    suites.append( test_image.test_suite(level=level) )
    """
    import unittest
    total_suite = unittest.TestSuite(suites)
    return total_suite

    
if __name__ == "__main__":
    test(10)
}
