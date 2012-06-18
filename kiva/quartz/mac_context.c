/*
 * mac_context.c
 *
 * Python extension to grab a CGContextRef from an NSView *
 *
 */

#include "Python.h"
#include "mac_context.h"

PyObject* get_mac_context(PyObject *self, PyObject *args)
{
    const char err_string[] = "get_mac_context() requires a pointer to an NSView.";
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
        if (tmp == 0)
        {
            PyErr_SetString(PyExc_ValueError, err_string);
            return NULL;
        }
        return Py_BuildValue("n", tmp);
    }
}

static PyMethodDef mac_context_methods[] = {
    {"get_mac_context", get_mac_context, METH_VARARGS,
        "get_mac_context(view_pointer) -> Returns the pointer (as an ssize_t) of the CGContextRef of an NSView pointer"},
    {NULL, NULL}
};

void initmac_context(void)
{
    Py_InitModule("mac_context", mac_context_methods);
}
