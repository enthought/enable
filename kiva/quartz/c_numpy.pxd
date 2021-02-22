# (C) Copyright 2004-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!

cdef extern from "numpy/arrayobject.h":

    cdef enum NPY_TYPES:
        NPY_BOOL
        NPY_BYTE
        NPY_UBYTE
        NPY_SHORT
        NPY_USHORT 
        NPY_INT
        NPY_UINT 
        NPY_LONG
        NPY_ULONG
        NPY_LONGLONG
        NPY_ULONGLONG
        NPY_FLOAT
        NPY_DOUBLE 
        NPY_LONGDOUBLE
        NPY_CFLOAT
        NPY_CDOUBLE
        NPY_CLONGDOUBLE
        NPY_OBJECT
        NPY_STRING
        NPY_UNICODE
        NPY_VOID
        NPY_NTYPES
        NPY_NOTYPE

    cdef enum requirements:
        NPY_CONTIGUOUS
        NPY_FORTRAN
        NPY_OWNDATA
        NPY_FORCECAST
        NPY_ENSURECOPY
        NPY_ENSUREARRAY
        NPY_ELEMENTSTRIDES
        NPY_ALIGNED
        NPY_NOTSWAPPED
        NPY_WRITEABLE
        NPY_UPDATEIFCOPY
        NPY_ARR_HAS_DESCR

        NPY_BEHAVED
        NPY_BEHAVED_NS
        NPY_CARRAY
        NPY_CARRAY_RO
        NPY_FARRAY
        NPY_FARRAY_RO
        NPY_DEFAULT

        NPY_IN_ARRAY
        NPY_OUT_ARRAY
        NPY_INOUT_ARRAY
        NPY_IN_FARRAY
        NPY_OUT_FARRAY
        NPY_INOUT_FARRAY

        NPY_UPDATE_ALL 

    cdef enum defines:
        # Note: as of Pyrex 0.9.5, enums are type-checked more strictly, so this
        # can't be used as an integer.
        NPY_MAXDIMS

    ctypedef struct npy_cdouble:
        double real
        double imag

    ctypedef struct npy_cfloat:
        double real
        double imag

    ctypedef int npy_intp 

    ctypedef extern class numpy.dtype [object PyArray_Descr]:
        cdef int type_num, elsize, alignment
        cdef char type, kind, byteorder, hasobject
        cdef object fields, typeobj

    ctypedef extern class numpy.ndarray [object PyArrayObject]:
        cdef char *data
        cdef int nd
        cdef npy_intp *dimensions
        cdef npy_intp *strides
        cdef object base
        cdef dtype descr
        cdef int flags

    ctypedef extern class numpy.flatiter [object PyArrayIterObject]:
        cdef int  nd_m1
        cdef npy_intp index, size
        cdef ndarray ao
        cdef char *dataptr
        
    ctypedef extern class numpy.broadcast [object PyArrayMultiIterObject]:
        cdef int numiter
        cdef npy_intp size, index
        cdef int nd
        # These next two should be arrays of [NPY_MAXITER], but that is
        # difficult to cleanly specify in Pyrex. Fortunately, it doesn't matter.
        cdef npy_intp *dimensions
        cdef void **iters

    object PyArray_ZEROS(int ndims, npy_intp* dims, NPY_TYPES type_num, int fortran)
    object PyArray_EMPTY(int ndims, npy_intp* dims, NPY_TYPES type_num, int fortran)
    dtype PyArray_DescrFromTypeNum(NPY_TYPES type_num)
    object PyArray_SimpleNew(int ndims, npy_intp* dims, NPY_TYPES type_num)
    int PyArray_Check(object obj)
    object PyArray_ContiguousFromAny(object obj, NPY_TYPES type, 
        int mindim, int maxdim)
    npy_intp PyArray_SIZE(ndarray arr)
    npy_intp PyArray_NBYTES(ndarray arr)
    void *PyArray_DATA(ndarray arr)
    object PyArray_FromAny(object obj, dtype newtype, int mindim, int maxdim,
		    int requirements, object context)
    object PyArray_FROMANY(object obj, NPY_TYPES type_num, int min,
                           int max, int requirements)
    object PyArray_NewFromDescr(object subtype, dtype newtype, int nd,
                                npy_intp* dims, npy_intp* strides, void* data,
                                int flags, object parent)
    void* PyArray_GETPTR2(object obj, int i, int j)

    void PyArray_ITER_NEXT(flatiter it)

    void import_array()
