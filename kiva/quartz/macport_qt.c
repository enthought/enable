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
    const char err_string[] = "get_macport() requires a pointer to an NSView.";    
    Py_ssize_t win_id_addr = 0;
    int err = 0;

    err = PyArg_ParseTuple(args, "n", &win_id_addr);
    if (err != 1)
    {
        PyErr_SetString(PyExc_ValueError, err_string);
        return NULL;
    }
    else
    {
        Py_ssize_t tmp = (Py_ssize_t)get_cg_context_ref((void *)win_id_addr);
        return Py_BuildValue("n", tmp);
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
