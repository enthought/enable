// (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
// All rights reserved.
//
// This software is provided without warranty under the terms of the BSD
// license included in LICENSE.txt and may be redistributed only under
// the conditions described in the aforementioned license. The license
// is also available online at http://www.enthought.com/licenses/BSD.txt
//
// Thanks for using Enthought open source!
%{
#include "kiva_gl_rect.h"
%}

%typemap(in) (kiva_gl::rect_type &rect)
{
    PyArrayObject* ary=NULL;
    int is_new_object;
    ary = obj_to_array_contiguous_allow_conversion($input, PyArray_DOUBLE,
                                                   is_new_object);

    int size[1] = {4};
    if (!ary ||
        !require_dimensions(ary, 1) ||
        !require_size(ary, size, 1))
    {
        goto fail;
    }

    double* data = (double*)(ary->data);
    kiva_gl::rect_type rect(data[0], data[1], data[2], data[3]);
    $1 = &rect;

    if (is_new_object)
    {
        Py_DECREF(ary);
    }
}

%typemap(out) kiva_gl::rect_type
{
    PyObject *pt = PyTuple_New(4);
    PyTuple_SetItem(pt,0,PyFloat_FromDouble($1.x));
    PyTuple_SetItem(pt,1,PyFloat_FromDouble($1.y));
    PyTuple_SetItem(pt,2,PyFloat_FromDouble($1.w));
    PyTuple_SetItem(pt,3,PyFloat_FromDouble($1.h));
    $result = pt;
}
