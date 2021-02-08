// (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
// All rights reserved.
//
// This software is provided without warranty under the terms of the BSD
// license included in LICENSE.txt and may be redistributed only under
// the conditions described in the aforementioned license. The license
// is also available online at http://www.enthought.com/licenses/BSD.txt
//
// Thanks for using Enthought open source!

// --------------------------------------------------------------------------
//
// Convert kiva_gl_agg::rgba types to/from Numeric arrays.  The rgba_as_array
// typemap will accept any 3 or 4 element sequence of float compatible
// objects and convert them into an kiva_gl_agg::rgba object.
//
// The typemap also converts any rgba output value back to a numeric array
// in python.  This is a more useful representation for numerical
// manipulation.
//
// --------------------------------------------------------------------------

%{
    #include "agg_color_rgba.h"
%}

%include "numpy.i"

#ifdef SWIGPYTHON

%typemap(in) rgba_as_array (int must_free=0)
{
  must_free = 0;
  if ((SWIG_ConvertPtr($input,(void **) &$1, SWIGTYPE_p_kiva_gl_agg__rgba,
                       SWIG_POINTER_EXCEPTION | 0 )) == -1)
  {
      PyErr_Clear();
      if (!PySequence_Check($input))
      {
          PyErr_SetString(PyExc_TypeError,"Expecting a sequence");
          return NULL;
      }

      int seq_len = PyObject_Length($input);
      if (seq_len != 3 && seq_len != 4)
      {
          PyErr_SetString(PyExc_ValueError,
                          "Expecting a sequence with 3 or 4 elements");
          return NULL;
      }

      double temp[4] = {0.0,0.0,0.0,1.0};
      for (int i =0; i < seq_len; i++)
      {
          PyObject *o = PySequence_GetItem($input,i);
          if (PyFloat_Check(o))
          {
             temp[i] = PyFloat_AsDouble(o);
          }
          else
          {
             PyObject* converted = PyNumber_Float(o);
             if (!converted)
             {
                 PyErr_SetString(PyExc_TypeError,
                                 "Expecting a sequence of floats");
                 return NULL;
             }
             temp[i] = PyFloat_AsDouble(converted);
             Py_DECREF(converted);
          }
          if ((temp[i] < 0.0) || (temp [i] > 1.0))
          {
              PyErr_SetString(PyExc_ValueError,
                              "Color values must be between 0.0 an 1.0");
              return NULL;
          }
      }
      $1 = new kiva_gl_agg::rgba(temp[0],temp[1],temp[2],temp[3]);
      must_free = 1;
   }
}

%typemap(freearg) rgba_as_array {
   if (must_free$argnum)
    delete $1;
}

%typemap(out) rgba_as_array
{
    npy_intp size = 4;
    $result = PyArray_SimpleNew(1, &size, PyArray_DOUBLE);
    double* data = (double*)((PyArrayObject*)$result)->data;
    data[0] = $1->r;
    data[1] = $1->g;
    data[2] = $1->b;
    data[3] = $1->a;
}

#endif

