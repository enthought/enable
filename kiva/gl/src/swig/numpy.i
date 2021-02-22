/* -*- c -*- */
// (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
// All rights reserved.
//
// This software is provided without warranty under the terms of the BSD
// license included in LICENSE.txt and may be redistributed only under
// the conditions described in the aforementioned license. The license
// is also available online at http://www.enthought.com/licenses/BSD.txt
//
// Thanks for using Enthought open source!

/* Set the input argument to point to a temporary variable */

/*
Here are the typemap helper functions for numpy arrays:

    PyArrayObject* obj_to_array_no_conversion(PyObject* input, int typecode)
    PyArrayObject* obj_to_array_allow_conversion(PyObject* input, int typecode,
                                                 int& is_new_object)
    PyArrayObject* obj_to_array_contiguous_allow_conversion(PyObject* input,
                                                        int typecode,
                                                        int& is_new_object)
    PyArrayObject* make_contiguous(PyArrayObject* ary, int& is_new_object,
                                   int min_dims = 0, int max_dims = 0)
    int require_contiguous(PyArrayObject* ary)
    int require_last_dimensions_contiguous(PyArrayObject* ary, int dim_count)
    int require_dimensions(PyArrayObject* ary, int exact_dimensions)
    int require_dimensions(PyArrayObject* ary, int* exact_dimensions, int n)
    int require_size(PyArrayObject* ary, int* size, int n)
*/

%{
#include "numpy/arrayobject.h"
#include <string>

#define is_array(a) ((a) && PyArray_Check((PyArrayObject *)a))
#define array_type(a) (int)(((PyArrayObject *)a)->descr->type_num)
#define array_dimensions(a) (((PyArrayObject *)a)->nd)
#define array_size(a,i) (((PyArrayObject *)a)->dimensions[i])
#define array_is_contiguous(a) (PyArray_ISCONTIGUOUS(ary))

std::string pytype_string(PyObject* py_obj)
{
    if(py_obj == NULL) return "C NULL value";
    if(PyCallable_Check(py_obj)) return "callable";
    if(PyString_Check(py_obj)) return "string";
    if(PyInt_Check(py_obj)) return "int";
    if(PyFloat_Check(py_obj)) return "float";
    if(PyDict_Check(py_obj)) return "dict";
    if(PyList_Check(py_obj)) return "list";
    if(PyTuple_Check(py_obj)) return "tuple";
    /*if(PyFile_Check(py_obj)) return "file";*/
    if(PyModule_Check(py_obj)) return "module";

    //should probably do more intergation (and thinking) on these.
    /*if(PyCallable_Check(py_obj) && PyInstance_Check(py_obj)) return "callable";
    if(PyInstance_Check(py_obj)) return "instance";*/
    if(PyCallable_Check(py_obj)) return "callable";
    return "unkown type";
}

std::string typecode_string(int typecode)
{
    std::string type_names[20] = {"char", "unsigned byte", "byte", "short",
                                  "unsigned short", "int", "unsigned int",
                                  "long", "float", "double", "complex float",
                                  "complex double", "object", "ntype",
                                  "unknown"};
    return type_names[typecode];
}

int type_match(int actual_type, int desired_type)
{
    int match;
    // Make sure input has correct numpy type. Allow character and byte to
    // match also allow int and long to match.
    if (actual_type != desired_type &&
            !(desired_type == PyArray_INT  && actual_type == PyArray_LONG) &&
            !(desired_type == PyArray_LONG && actual_type == PyArray_INT))
    {
        match = 0;
    }
    else
    {
        match = 1;
    }
    return match;
}

PyArrayObject* obj_to_array_no_conversion(PyObject* input, int typecode)
{
    PyArrayObject* ary = NULL;
    if (is_array(input) && array_type(input) == typecode)
    {
        ary = (PyArrayObject*) input;
    }
    else if is_array(input)
    {
        char msg[255] = "Array of type '%s' required.  Array of type '%s' given";
        std::string desired_type = typecode_string(typecode);
        std::string actual_type = typecode_string(array_type(input));
        PyErr_Format(PyExc_TypeError, msg,
                     desired_type.c_str(), actual_type.c_str());
        ary = NULL;
    }
    else
    {
        char msg[255] = "Array of type '%s' required.  A %s was given";
        std::string desired_type = typecode_string(typecode);
        std::string actual_type = pytype_string(input);
        PyErr_Format(PyExc_TypeError, msg,
                     desired_type.c_str(), actual_type.c_str());
        ary = NULL;
    }
    return ary;
}

PyArrayObject* obj_to_array_allow_conversion(PyObject* input, int typecode,
                                             int& is_new_object)
{
    // Convert object to a numpy array with the given typecode.
    //
    // Return:
    //   On Success, return a valid PyArrayObject* with the correct type.
    //   On failure, return NULL.  A python error will have been set.

    PyArrayObject* ary = NULL;
    if (is_array(input) && type_match(array_type(input),typecode))
    {
        ary = (PyArrayObject*) input;
        is_new_object = 0;
    }
    else
    {
        PyObject* py_obj = PyArray_FromObject(input, typecode, 0, 0);
        // If NULL, PyArray_FromObject will have set python error value.
        ary = (PyArrayObject*) py_obj;
        is_new_object = 1;
    }

    return ary;
}

PyArrayObject* make_contiguous(PyArrayObject* ary, int& is_new_object,
                               int min_dims = 0, int max_dims = 0)
{
    PyArrayObject* result;
    if (array_is_contiguous(ary))
    {
        result = ary;
        is_new_object = 0;
    }
    else
    {
        result = (PyArrayObject*) PyArray_ContiguousFromObject(
                                                          (PyObject*)ary,
                                                          array_type(ary),
                                                          min_dims,
                                                          max_dims);
        is_new_object = 1;
    }

    return result;
}

PyArrayObject* obj_to_array_contiguous_allow_conversion(PyObject* input,
                                                        int typecode,
                                                        int& is_new_object)
{
    int is_new1 = 0;
    int is_new2 = 0;
    PyArrayObject* ary1 = obj_to_array_allow_conversion(input, typecode,
                                                        is_new1);
    if (ary1)
    {
        PyArrayObject* ary2 = make_contiguous(ary1, is_new2);

        if ( is_new1 && is_new2)
        {
            Py_DECREF(ary1);
        }
        ary1 = ary2;
    }

    is_new_object = is_new1 || is_new2;
    return ary1;
}


int require_contiguous(PyArrayObject* ary)
{
    // Test whether a python object is contiguous.
    //
    // Return:
    //     1 if array is contiguous.
    //     Otherwise, return 0 and set python exception.
    int contiguous = 1;
    if (!array_is_contiguous(ary))
    {
        char msg[255] = "Array must be contiguous.  A discontiguous array was given";
        PyErr_SetString(PyExc_TypeError, msg);
        contiguous = 0;
    }
    return contiguous;
}

// Useful for allowing images with discontiguous first dimension.
// This sort of array is used for arrays mapped to Windows bitmaps.
/*
int require_last_dimensions_contiguous(PyArrayObject* ary, int dim_count)
{
    int contiguous = 1;
    if (array_is_contiguous(ary))
    {
        char msg[255] = "Array must be contiguous.  A discontiguous array was given";
        PyErr_SetString(PyExc_TypeError, msg);
        contiguous = 0;
    }
    return contiguous;
}
*/

int require_dimensions(PyArrayObject* ary, int exact_dimensions)
{
    int success = 1;
    if (array_dimensions(ary) != exact_dimensions)
    {
        char msg[255] = "Array must be have %d dimensions.  Given array has %d dimensions";
        PyErr_Format(PyExc_TypeError, msg,
                     exact_dimensions, array_dimensions(ary));
        success = 0;
    }
    return success;
}

int require_dimensions(PyArrayObject* ary, int* exact_dimensions, int n)
{
    int success = 0;
    int i;
    for (i = 0; i < n && !success; i++)
    {
        if (array_dimensions(ary) == exact_dimensions[i])
        {
            success = 1;
        }
    }
    if (!success)
    {
        char dims_str[255] = "";
        char s[255];
        for (int i = 0; i < n-1; i++)
        {
             sprintf(s, "%d, ", exact_dimensions[i]);
             strcat(dims_str,s);
        }
        sprintf(s, " or %d", exact_dimensions[n-1]);
        strcat(dims_str,s);
        char msg[255] = "Array must be have %s dimensions.  Given array has %d dimensions";
        PyErr_Format(PyExc_TypeError, msg, dims_str, array_dimensions(ary));
    }
    return success;
}

int require_size(PyArrayObject* ary, int* size, int n)

{
    int i;
    int success = 1;
    for(i=0; i < n;i++)
    {
        if (size[i] != -1 &&  size[i] != array_size(ary,i))
        {
           success = 0;
        }
    }

    if (!success)
    {
        int len;
        char desired_dims[255] = "[";
        char s[255];
        for (i = 0; i < n; i++)
        {
            if (size[i] == -1)
            {
                sprintf(s, "*,");
            }
            else
            {
                sprintf(s, "%d,", size[i]);
            }
            strcat(desired_dims,s);
        }
        len = strlen(desired_dims);
        desired_dims[len-1] = ']';

        char actual_dims[255] = "[";
        for (i = 0; i < n; i++)
        {
            sprintf(s, "%d,", (int)array_size(ary,i));
            strcat(actual_dims,s);
        }
        len = strlen(actual_dims);
        actual_dims[len-1] = ']';

        char msg[255] = "Array must be have shape of %s.  Given array has shape of %s";
        PyErr_Format(PyExc_TypeError, msg, desired_dims, actual_dims);
    }
    return success;
}


%}

%pythoncode %{
from numpy import ndarray

def is_array(obj):
    return type(obj) is ndarray

def is_correct_type(obj, numpy_type):
    return is_array(obj) and (obj.dtype == numpy_type)

def numpy_check(obj, typecode,
                exact_size = [],
                must_be_contiguous = 1,
                allow_coersion = 0):

    if is_correct_type(obj, typecode):
        ary = obj
    elif allow_coersion:
        ary = asarray(obj,typecode)
    else:
        raise TypeError("input is not an array or the array has the wrong type")

    if must_be_contiguous and not ary.flags["CONTIGUOUS"]:
        if allow_coersion:
            ary = ary.copy()
        else:
            raise TypeError("input array must be contiguous")

    # check number of dimensions
    required_dims = len(exact_size)
    if required_dims and required_dims != len(ary.shape):
        raise ValueError("The input array does not have the correct shape")

    # check exact shape of each dimension
    cnt = 0
    for desired,actual in zip(exact_size,ary.shape):
        if desired != -1 and desired != actual:
            raise ValueError("The %d dimensions of the array has the wrong shape" % (cnt))
        cnt += 1

    return ary
%}

%init %{
    Py_Initialize();
    import_array();
    PyImport_ImportModule("numpy");
%}
