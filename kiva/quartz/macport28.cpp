// (C) Copyright 2004-2021 Enthought, Inc., Austin, TX
// All rights reserved.
//
// This software is provided without warranty under the terms of the BSD
// license included in LICENSE.txt and may be redistributed only under
// the conditions described in the aforementioned license. The license
// is also available online at http://www.enthought.com/licenses/BSD.txt
//
// Thanks for using Enthought open source!

#include "wx/wx.h"

#include "Python.h"

extern "C"
{
    void initmacport(void);
}

/* This converts a string of hex digits into an unsigned long.  It reads 
   This code is a modified version of SWIG_UnpackData from weave/swigptr2.py,
   and which I believe originated from the SWIG sources. */
unsigned long hexstr_to_long(const char *c, char bytesize = 4) {
    unsigned long retval = 0;
    
    unsigned char *u = reinterpret_cast<unsigned char*>(&retval);
    const unsigned char *eu =  u + bytesize;
    for (; u != eu; ++u) {
        register int d = *(c++);
        register unsigned char uu = 0;
        if ((d >= '0') && (d <= '9'))
            uu = ((d - '0') << 4);
        else if ((d >= 'a') && (d <= 'f'))
            uu = ((d - ('a'-10)) << 4);
        else 
            return 0;
        d = *(c++);
        if ((d >= '0') && (d <= '9'))
            uu |= (d - '0');
        else if ((d >= 'a') && (d <= 'f'))
            uu |= (d - ('a'-10));
        else 
            return 0;
        *u = uu;
    }
    return retval;
}

PyObject* get_macport(PyObject *self, PyObject *args)
{
    const char err_string[] = "get_macport() requires a SWIG 'this' string.";
    
    // the string representing the address embedded in the SWIG this ptr
    char *dc_addr_str = NULL;
    int length = 0;
    int err = 0;
    wxDC *p_dc = NULL;
    wxGraphicsContext *p_gc = NULL;

    err = PyArg_ParseTuple(args, "s#", &dc_addr_str, &length);
    if (err != 1)
    {
        PyErr_SetString(PyExc_ValueError, err_string);
        return NULL;
    }
    else if (length < 10)
    {
        PyErr_SetString(PyExc_ValueError, err_string);
        return NULL;
    }
    else
    {
        p_dc = reinterpret_cast<wxDC*>(hexstr_to_long(dc_addr_str+1, 4));
        p_gc = reinterpret_cast<wxGraphicsContext*>(p_dc->GetGraphicsContext());
        unsigned long tmp = reinterpret_cast<unsigned long>(p_gc->GetNativeContext());
        return Py_BuildValue("k", tmp);
    }
}

static PyMethodDef macport_methods[] = {
    {"get_macport", get_macport, METH_VARARGS,
        "get_macport(dc.this) -> Returns the pointer (as an unsigned long) of the CGContextRef of a wxDC.this SWIG pointer"},
    {NULL, NULL}
};

void initmacport(void)
{
    Py_InitModule("macport", macport_methods);
}
