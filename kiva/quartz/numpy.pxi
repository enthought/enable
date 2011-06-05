# :Author:    Robert Kern
# :Copyright: 2004, Enthought, Inc.
# :License:   BSD Style


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
