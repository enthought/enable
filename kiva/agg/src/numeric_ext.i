// (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
// All rights reserved.
//
// This software is provided without warranty under the terms of the BSD
// license included in LICENSE.txt and may be redistributed only under
// the conditions described in the aforementioned license. The license
// is also available online at http://www.enthought.com/licenses/BSD.txt
//
// Thanks for using Enthought open source!
%include "numeric.i"

%{
#ifndef NUMPY
#include "Numeric/arrayobject.h"

/* This is the basic array allocation function. */
PyObject *PyArray_FromDimsAndStridesAndDataAndDescr(int nd, int *d, int* st,
                                                    PyArray_Descr *descr,
                                                    char *data) 
{
    PyArrayObject *self;
    int i,sd;
    int *dimensions, *strides;
    int flags= CONTIGUOUS | OWN_DIMENSIONS | OWN_STRIDES;
    
    //static int calls = 0;
    //calls++; 
    //printf("allocs: %d\n;", calls);
    
    dimensions = strides = NULL;
        
    if (nd < 0) 
    {
        PyErr_SetString(PyExc_ValueError, 
                        "number of dimensions must be >= 0");
        return NULL;
    }
        
    if (nd > 0) 
    {
        if ((dimensions = (int *)malloc(nd*sizeof(int))) == NULL) {
            PyErr_SetString(PyExc_MemoryError, "can't allocate memory for array");
            goto fail;
        }
        if ((strides = (int *)malloc(nd*sizeof(int))) == NULL) {
            PyErr_SetString(PyExc_MemoryError, "can't allocate memory for array");
            goto fail;
        }
        memmove(dimensions, d, sizeof(int)*nd);
        memmove(strides, st, sizeof(int)*nd);
    }
        
    // This test for continguity
    sd = descr->elsize;
    for(i=nd-1;i>=0;i--) 
    {
        if (strides[i] <= 0) 
        {
            PyErr_SetString(PyExc_ValueError, "strides must be positive");
            goto fail;
        }
        if (dimensions[i] <= 0) 
        {
            /* only allow positive dimensions in this function */
            PyErr_SetString(PyExc_ValueError, "dimensions must be positive");
            goto fail;
        }
        if (strides[i] != sd)
        {
            flags &= !CONTIGUOUS;
        }    
        /* 
           This may waste some space, but it seems to be
           (unsuprisingly) unhealthy to allow strides that are
           longer than sd.
        */
        sd *= dimensions[i] ? dimensions[i] : 1;
    }
        
    /* Make sure we're alligned on ints. */
    sd += sizeof(int) - sd%sizeof(int); 
        
    if (data == NULL) 
    {
        if ((data = (char *)malloc(sd)) == NULL) 
        {
            PyErr_SetString(PyExc_MemoryError, "can't allocate memory for array");
            goto fail;
        }
        flags |= OWN_DATA;
    }

    if((self = PyObject_NEW(PyArrayObject, &PyArray_Type)) == NULL) 
        goto fail;
    if (flags & OWN_DATA) 
        memset(data, 0, sd);
        
    self->data=data;
    self->dimensions = dimensions;
    self->strides = strides;
    self->nd=nd;
    self->descr=descr;
    self->base = (PyObject *)NULL;
    self->flags = flags;

/* Numeric versions prior to 23.3 do not have the weakreflist field. 
   By default we include it.  You must explicitly define:
    NUMERIC_DOES_NOT_HAVE_WEAKREF    
*/
#ifndef NUMERIC_DOES_NOT_HAVE_WEAKREF    
    self->weakreflist = (PyObject *)NULL;
#endif

    return (PyObject*)self;
        
 fail:        
    if (flags & OWN_DATA) free(data);
    if (dimensions != NULL) free(dimensions);
    if (strides != NULL) free(strides);
    return NULL;    
}
#endif

%}