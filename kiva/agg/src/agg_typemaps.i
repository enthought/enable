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
// Generic typemap to handle enumerated types.
//
// Both agg and kiva have quite a few enumerated types.  SWIG will wrap
// functions that use these as arguments to require a pointer to an
// enumerated type object.  This isn't very convenient.  It is much nicer
// to just pass an integer.  this generic converter can be used in to
// allow this.
//
// To apply it to a type, for example agg24::marker_e, do the following
//
//      %apply(kiva_enum_typemap) { agg24::marker_e }
//
// Now any function that expects a marker_e will accept integer values as
// input.
// --------------------------------------------------------------------------

%include "numeric.i"

%typemap(in) kiva_enum_typemap type {

   int temp = PyInt_AsLong($input);
   if (PyErr_Occurred()) SWIG_fail;
   $1 = $1_ltype(temp);
}


// --------------------------------------------------------------------------
// Typemaps for (double x, double y) points
//
//    For: *
//    
//    This is useful for places where ints may be passed in and need to
//    be converted. Python 2.6 requires this
// --------------------------------------------------------------------------
%typemap(in) double x, double y
{
    if (PyNumber_Check($input))
    {
        $1 = static_cast<double>(PyFloat_AsDouble($input));
    }
    else
    {
        SWIG_exception(SWIG_TypeError, "Expected argument $argnum of type '$1_type'");
     }
}

// --------------------------------------------------------------------------
// Typemaps for (double* pts, int Npts) used in lines()
//
//    For: compiled_path and graphics_context
//
//    This typemap takes any Nx2 input (nested sequences or an Nx2 array). If
//    the input has the wrong shape or can't be converted to a double, an
//    exception is raised.  It is more efficient if the input passed in is a
//    contiguous array, so if you're calling lines(pts) a lot of times, make
//    pts an array.
//
// --------------------------------------------------------------------------

%typemap(in) (double* point_array, int point_count) (PyArrayObject* ary=NULL,
                                         int is_new_object)
{
    ary = obj_to_array_contiguous_allow_conversion($input, PyArray_DOUBLE,
                                                   is_new_object);
    int size[2] = {-1,2};
    if (!ary ||
        !require_dimensions(ary,2) ||
        !require_size(ary,size,2))
    {
        goto fail;
    }
    $1 = (double*) ary->data;
    $2 = ary->dimensions[0];
}

%typemap(freearg) (double* point_array, int point_count)
{
    if (is_new_object$argnum)
    {
        Py_XDECREF(ary$argnum);
    }
}

// --------------------------------------------------------------------------
// Typemaps for (unsigned char* results, int Nresults)
//
//    For: points_in_polygon
//
//    This typemap takes any N input.
//
// --------------------------------------------------------------------------

%typemap(in) (unsigned char* results, int Nresults) (PyArrayObject* ary=NULL,
                                           int is_new_object)
{
    ary = obj_to_array_contiguous_allow_conversion($input, PyArray_BOOL,
                                                   is_new_object);
    int size[1] = {-1};
    if (!ary ||
        !require_dimensions(ary,1) ||
        !require_size(ary,size,1))
    {
        goto fail;
    }
    $1 = (unsigned char*) ary->data;
    $2 = ary->dimensions[0];
}

%typemap(freearg) (unsigned char* results, int Nresults)
{
    if (is_new_object$argnum)
    {
        Py_XDECREF(ary$argnum);
    }
}


/* Typemaps for rects(double* all_rects, int Nrects)

    For: compiled_path and graphics_context

    This typemap takes any Nx4 input (nested sequences or an Nx4 array). If
    the input has the wrong shape or can't be converted to a double, an
    exception is raised.  It is more efficient if the input passed in is a
    contiguous array, so if you're calling rects(all_rects) a lot of times,
    make all_rects an array.
*/
%typemap(in)  (double* rect_array, int rect_count) (PyArrayObject* ary=NULL,
                                                    int is_new_object)
{
    ary = obj_to_array_contiguous_allow_conversion($input, PyArray_DOUBLE,
                                                   is_new_object);
    int size[2] = {-1,4};
    if (!ary ||
        !require_dimensions(ary,2) ||
        !require_size(ary,size,2))
    {
        goto fail;
    }
    $1 = (double*) ary->data;
    $2 = ary->dimensions[0];
}

%typemap(freearg) (double* rect_array, int rect_count)
{
    if (is_new_object$argnum)
    {
        Py_XDECREF(ary$argnum);
    }
}

// --------------------------------------------------------------------------
//
// vertex() returns ( pt, cmd) where pt is a tuple (x,y)
//
// This tells SWIG to treat an double * argument with name 'x' as
// an output value.  We'll append the value to the current result which
// is guaranteed to be a List object by SWIG.
// --------------------------------------------------------------------------
%typemap(in,numinputs=0) (double *vertex_x, double* vertex_y)(double temp1,
                                                              double temp2)
{
    temp1 = 0; $1 = &temp1;
    temp2 = 0; $2 = &temp2;
}

%typemap(argout) (double *vertex_x, double* vertex_y)
{
    PyObject *px = PyFloat_FromDouble(*$1);
    PyObject *py = PyFloat_FromDouble(*$2);
    PyObject *pt = PyTuple_New(2);
    PyTuple_SetItem(pt,0,px);
    PyTuple_SetItem(pt,1,py);
    PyObject *return_val = PyTuple_New(2);
    PyTuple_SetItem(return_val,0,pt);
    // result is what was returned from vertex
    PyTuple_SetItem(return_val,1,$result);
    //Py_DECREF($result);
    $result = return_val;
}

// --------------------------------------------------------------------------
// map to output arguments into a 2-tuple
// --------------------------------------------------------------------------
%typemap(in,numinputs=0) (double *pt_x, double* pt_y)(double temp1,
                                                      double temp2)
{
    temp1 = 0; $1 = &temp1;
    temp2 = 0; $2 = &temp2;
}
%typemap(argout) (double *pt_x, double *pt_y)
{
    PyObject *px = PyFloat_FromDouble(*$1);
    PyObject *py = PyFloat_FromDouble(*$2);
    PyObject *pt = PyTuple_New(2);
    PyTuple_SetItem(pt,0,px);
    PyTuple_SetItem(pt,1,py);
    //Py_DECREF($result);
    $result = pt;
}

// --------------------------------------------------------------------------
// map an 6 element double* output into a Numeric array.
// --------------------------------------------------------------------------
%typemap(in, numinputs=0) double *array6 (double temp[6]) {
   $1 = temp;
}

%typemap(argout) double *array6 {
   // Append output value $1 to $result
   npy_intp dims = 6;
   PyArrayObject* ary_obj = (PyArrayObject*) PyArray_SimpleNew(1,&dims,PyArray_DOUBLE);
   if( ary_obj == NULL )
    return NULL;
   double* data = (double*)ary_obj->data;
   for (int i=0; i < 6;i++)
       data[i] = $1[i];
   Py_DECREF($result);
   $result = PyArray_Return(ary_obj);
}

// --------------------------------------------------------------------------
// Typemaps for graphics_context.set_line_dash()
//
//    For:
//
//    This typemap takes None or any N element input (sequence or array). If
//    the input is None, it passes a 2 element array of zeros to in as the
//    pattern. If the input is a sequence and isn't 1D or can't be converted
//    to a double, an exception is raised.
// --------------------------------------------------------------------------

%typemap(in) (double* dash_pattern, int n) (PyArrayObject* ary=NULL,
                                            int is_new_object,
                                            double temp[2])
{
    is_new_object = 0;
    if ($input == Py_None)
    {
        temp[0] = 0.0;
        temp[1] = 0.0;
        $1 = temp;
        $2 = 2;
    }
    else
    {
        ary = obj_to_array_contiguous_allow_conversion($input, PyArray_DOUBLE,
                                                       is_new_object);
        if (!ary ||
            !require_dimensions(ary,1))
        {
            goto fail;
        }
        $1 = (double*) ary->data;
        $2 = ary->dimensions[0];
    }
}

%typemap(freearg) (double* dash_pattern, int n)
{
    if (is_new_object$argnum)
    {
        Py_XDECREF(ary$argnum);
    }
}

// --------------------------------------------------------------------------
// Image typemaps
//
//    Currently, this requires a contiguous array.  It should be fixed to
//    allow arrays that are only contiguous along the last two dimensions.
//    This is because the windows bitmap format requires that each row of
//    pixels (scan line) is word aligned (16 bit boundaries). As a result, rgb
//    images compatible with this format potentially
//    need a pad byte at the end of each scanline.
// --------------------------------------------------------------------------

%typemap(in) (unsigned char *image_data=NULL, int width, int height, int stride)
{
    PyArrayObject* ary = obj_to_array_no_conversion($input, PyArray_UBYTE);
    int dimensions[2] = {2,3};
// !! No longer requiring contiguity because some bitmaps are padded at the
// !! end (i.e. Windows).  We should probably special case that one though,
// !! and re-instate the contiguous policy...
//    if (!ary ||
//        !require_dimensions(ary,dimensions,2) ||
//        !require_contiguous(ary))
    if (!ary ||
        !require_dimensions(ary,dimensions,2))
    {
        goto fail;
    }
    $1 = (unsigned char*) ary->data;
    // notice reversed orders...
    $2 = ary->dimensions[1];
    $3 = ary->dimensions[0];
    $4 = ary->strides[0];
}

// --------------------------------------------------------------------------
// Some functions create new objects and return these to python.  By
// default, SWIG sets these objects as "unowned" by the shadow class
// created to represent them in python.  The result is that these objects
// are not freed when the shadow object calls its __del__ method.  Here
// the thisown flag is set to 1 so that the object will be destroyed on
// destruction.
// --------------------------------------------------------------------------

%typemap(out) owned_pointer
{
    $result = SWIG_NewPointerObj((void *) $1, $1_descriptor, 1);
}

//---------------------------------------------------------------------
// Gradient support
//---------------------------------------------------------------------
%typemap(in) std::vector<kiva::gradient_stop> (PyArrayObject* ary=NULL,
                                         		   int is_new_object)
{
    PyArrayObject* ary = obj_to_array_no_conversion($input, PyArray_DOUBLE);
    if (ary == NULL)
    {
        goto fail;
    }
    
    std::vector<kiva::gradient_stop> stops;
    
    for (int i = 0; i < ary->dimensions[0]; i++)
    {
        // the stop is offset, red, green, blue, alpha
        double* data = (double*)(ary->data);
        agg24::rgba8 color(data[5*i+1]*255, data[5*i+2]*255, data[5*i+3]*255, data[5*i+4]*255);
        stops.push_back(kiva::gradient_stop(data[5*i], color));
    }
    
    
    $1 = stops;
}

%typemap(in) const char* gradient_arg (PyObject *utfstr=NULL) {
  if (PyBytes_Check($input))
  {
    $1 = (char *)PyBytes_AsString($input);
  }
#if PY_VERSION_HEX >= 0x03030000
  else if (PyUnicode_Check($input))
  {
    $1 = (char *)PyUnicode_AsUTF8($input);
  }
#elif PY_VERSION_HEX < 0x03000000
  else if (PyUnicode_Check($input))
  {
    utfstr = PyUnicode_AsUTF8String($input);
    $1 = (char *)PyString_AsString(utfstr);
  }
#endif
  else
  {
    PyErr_SetString(PyExc_TypeError, "not a string");
    return NULL;
  }
}
%typemap(freearg) const char* gradient_arg
{
    if (utfstr$argnum)
    {
        Py_DECREF(utfstr$argnum);
    }
}
