// (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
// All rights reserved.
//
// This software is provided without warranty under the terms of the BSD
// license included in LICENSE.txt and may be redistributed only under
// the conditions described in the aforementioned license. The license
// is also available online at http://www.enthought.com/licenses/BSD.txt
//
// Thanks for using Enthought open source!

// Map a Python sequence into any sized C double array
// This handles arrays and sequences with non-float values correctly.
// !! Optimize for array conversion??

#ifdef SWIGPYTHON

%typemap(in) double[ANY](double temp[$1_dim0]) {
  int i;
  if (!PySequence_Check($input)) {
      PyErr_SetString(PyExc_TypeError,"Expecting a sequence");
      return NULL;
  }
  if (PyObject_Length($input) != $1_dim0) {
      PyErr_SetString(PyExc_ValueError,"Expecting a sequence with $1_dim0 elements");
      return NULL;
  }
  for (i =0; i < $1_dim0; i++) {
      PyObject *o = PySequence_GetItem($input,i);
      if (PyFloat_Check(o)) {
         temp[i] = PyFloat_AsDouble(o);
      }  
      else {
         PyObject* converted = PyNumber_Float(o);
         if (!converted) {
             PyErr_SetString(PyExc_TypeError,"Expecting a sequence of floats");
             return NULL;
         }
         temp[i] = PyFloat_AsDouble(converted);  
         Py_DECREF(converted);
      }
  }
  $1 = &temp[0];
}

#endif
