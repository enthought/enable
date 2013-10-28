%include "std_string.i"

// These typemaps are needed to handle member access to font_type.name
// and friends.  They really should by part of std_string.i, shouldn't
// they?
#ifdef SWIGPYTHON
%typemap(in) std::string * {
  if (PyBytes_Check ($input))
  {
    $1 = new std::string((char *)PyBytes_AsString($input));
  }
#if PY_VERSION_HEX >= 0x03030000
  else if (PyUnicode_Check($input))
  {
    $1 = new std::string((char *)PyUnicode_AsUTF8($input));
  }
#endif
  else
  {
    PyErr_SetString (PyExc_TypeError, "not a String");
    return NULL;
  }
}
%typemap(out) std::string * {
  $result = SWIG_Python_str_FromChar((const char *)$1->c_str());
}
%typemap(freearg) std::string * {
  if ($1)
  {
    delete $1;
  }
}

#endif   /* SWIGPYTHON */

