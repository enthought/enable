/*
 * macport_qt.c
 *
 * Python extension to grab a CGContextRef from an NSView *
 *
 */

#include "Python.h"
#include "macport_cocoa.h"

PyObject* get_macport(PyObject *self, PyObject *args)
{
    const char err_string[] = "get_macport() requires an int argument.";
    
    // the string representing the address embedded in the SWIG this ptr
    unsigned long win_id_addr = 0;
    int err = 0;

    err = PyArg_ParseTuple(args, "k", &win_id_addr);
    if (err != 1)
    {
        PyErr_SetString(PyExc_ValueError, err_string);
        return NULL;
    }
    else
    {
        unsigned long tmp = (unsigned long)get_cg_context_ref((void *)win_id_addr);
        return Py_BuildValue("k", tmp);
    }
}

static PyMethodDef macport_methods[] = {
    {"get_macport", get_macport, METH_VARARGS,
        "get_macport(winId) -> Returns the pointer (as an unsigned long) of the CGContextRef of a Qt winId pointer"},
    {NULL, NULL}
};

void initmacport_qt(void)
{
    Py_InitModule("macport_qt", macport_methods);
}
