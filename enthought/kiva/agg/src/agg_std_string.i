%include "std_string.i"

// These typemaps are needed to handle member access to font_type.name
// and friends.  They really should by part of std_string.i, shouldn't
// they?
#ifdef SWIGPYTHON
%typemap(in) std::string * {
  if (PyString_Check ($input))
  {
    $1 = new std::string((char *)PyString_AsString($input));
  }
  else
  {
    PyErr_SetString (PyExc_TypeError, "not a String");
    return NULL;
  }
}
%typemap(out) std::string * {
  $result = PyString_FromString((const char *)$1->c_str()); 
}
%typemap(freearg) std::string * {
  if ($1)
  {
    delete $1;
  }
}

#endif   /* SWIGPYTHON */

