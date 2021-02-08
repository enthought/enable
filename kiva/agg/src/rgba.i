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
#include "agg_color_rgba.h"
%}

%include "numeric.i"

%typemap(in, numinputs=0) double* out (double temp[4]) {
   $1 = temp;
}

%typemap(argout) double *out {
   // Append output value $1 to $result
   npy_intp dims = 4;
   PyArrayObject* ary_obj = (PyArrayObject*) PyArray_SimpleNew(1,&dims,PyArray_DOUBLE);
   if( ary_obj == NULL )
    return NULL;
   double* data = (double*)ary_obj->data;
   for (int i=0; i < 4;i++)
       data[i] = $1[i];
   Py_DECREF($result);
   $result = PyArray_Return(ary_obj);
}

%typemap(check) (double r) 
{
    if ($1 < 0.0 || $1 > 1.0)
    {
        PyErr_Format(PyExc_ValueError,
                     "color values must be between 0.0 and 1.0, Got: %g", $1);
    }
}

%apply (double r) {double g, double b, double a};  


namespace agg24
{
    %rename(_Rgba) rgba;
    struct rgba
    {
        double r;
        double g;
        double b;
        double a;

        rgba(double r_=0.0, double g_=0.0, double b_=0.0, double a_=1.0);
        //void opacity(double a_);
        //double opacity() const;
        rgba gradient(rgba c, double k) const;
        const rgba &premultiply();
    };
}

%extend agg24::rgba
{    
    char *__repr__()
    {
        static char tmp[1024];
        sprintf(tmp,"Rgba(%g,%g,%g,%g)", self->r,self->g,self->b,self->a);
        return tmp;
    }
    int __eq__(agg24::rgba& o)
    {
        return (self->r == o.r && self->g == o.g && 
                self->b == o.b && self->a == o.a);
    }
    void asarray(double* out)
    {
        out[0] = self->r;
        out[1] = self->g;
        out[2] = self->b;
        out[3] = self->a;    
    }
}


%pythoncode %{
def is_sequence(arg):
    try:
        len(arg)
        return 1
    except:
        return 0

# Use sub-class to allow sequence as input
class Rgba(_Rgba):
    def __init__(self,*args):
        if len(args) == 1 and is_sequence(args[0]):
            args = tuple(args[0])
            if len(args) not in [3,4]:
                raise ValueError("array argument must be 1x3 or 1x4")
        _Rgba.__init__(self,*args)
%}

%clear double r, double g, double b, double a;
