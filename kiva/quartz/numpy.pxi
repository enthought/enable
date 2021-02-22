# (C) Copyright 2004-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!

cdef extern from "numpy/oldnumeric.h":
    ctypedef enum PyArray_TYPES:
        PyArray_CHAR
        PyArray_UBYTE
        PyArray_SBYTE
        PyArray_SHORT
        PyArray_USHORT
        PyArray_INT
        PyArray_UINT
        PyArray_LONG
        PyArray_FLOAT
        PyArray_DOUBLE
        PyArray_CFLOAT
        PyArray_CDOUBLE
        PyArray_OBJECT
        PyArray_NTYPES
        PyArray_NOTYPE

    struct PyArray_Descr:
        int type_num, elsize
        char type

    ctypedef class numpy.ndarray [object PyArrayObject]:
        cdef char *data
        cdef int nd
        cdef int *dimensions
        cdef int *strides
        cdef object base
        cdef PyArray_Descr *descr
        cdef int flags

    ndarray PyArray_FromDims(int ndims, int* dims, int item_type)
    int PyArray_Free(object obj, char* data)

    void import_array()
