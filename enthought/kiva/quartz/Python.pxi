# :Author:    Robert Kern
# :Copyright: 2004, Enthought, Inc.
# :License:   BSD Style

ctypedef void (*cobject_destr)(void *)

cdef extern from "Python.h":
    ctypedef int size_t
    ctypedef int Py_ssize_t
    char* PyString_AsString(object string)
    object PyString_FromString(char* c_string)
    object PyString_FromStringAndSize(char* v, Py_ssize_t len)
    int PyString_AsStringAndSize(object obj, char **buffer, Py_ssize_t *length)
    int PyObject_AsCharBuffer(object obj, char **buffer, Py_ssize_t *buffer_len)
    int PyObject_AsReadBuffer(object obj, void **buffer, Py_ssize_t *buffer_len)
    int PyObject_CheckReadBuffer(object o)
    int PyObject_AsWriteBuffer(object obj, void **buffer, Py_ssize_t *buffer_len)

    void* PyMem_Malloc(size_t n)
    void* PyMem_Realloc(void* buf, size_t n)
    void PyMem_Free(void* buf)

    void Py_DECREF(object obj)
    void Py_XDECREF(object obj)
    void Py_INCREF(object obj)
    void Py_XINCREF(object obj)

    int PyUnicode_Check(ob)
    int PyString_Check(ob)

    ctypedef int Py_UNICODE
    Py_UNICODE *PyUnicode_AS_UNICODE(ob)
    int PyUnicode_GET_SIZE(ob)
    char *PyString_AS_STRING(ob)


    object PyCObject_FromVoidPtr(void* cobj, cobject_destr destr)
    void* PyCObject_AsVoidPtr(object self)

cdef extern from "string.h":
    void *memcpy(void *s1, void *s2, int n)

cdef extern from "math.h":
    double fabs(double x)

cdef extern from "stdlib.h":
    void free(void *ptr)
    void *malloc(size_t size)
    void *realloc(void *ptr, size_t size)

