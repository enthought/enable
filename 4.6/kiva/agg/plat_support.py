# This file was automatically generated by SWIG (http://www.swig.org).
# Version 3.0.11
#
# Do not make changes to this file unless you know what you are doing--modify
# the SWIG interface file instead.

from sys import version_info as _swig_python_version_info
if _swig_python_version_info >= (2, 7, 0):
    def swig_import_helper():
        import importlib
        pkg = __name__.rpartition('.')[0]
        mname = '.'.join((pkg, '_plat_support')).lstrip('.')
        try:
            return importlib.import_module(mname)
        except ImportError:
            return importlib.import_module('_plat_support')
    _plat_support = swig_import_helper()
    del swig_import_helper
elif _swig_python_version_info >= (2, 6, 0):
    def swig_import_helper():
        from os.path import dirname
        import imp
        fp = None
        try:
            fp, pathname, description = imp.find_module('_plat_support', [dirname(__file__)])
        except ImportError:
            import _plat_support
            return _plat_support
        try:
            _mod = imp.load_module('_plat_support', fp, pathname, description)
        finally:
            if fp is not None:
                fp.close()
        return _mod
    _plat_support = swig_import_helper()
    del swig_import_helper
else:
    import _plat_support
del _swig_python_version_info

try:
    _swig_property = property
except NameError:
    pass  # Python < 2.2 doesn't have 'property'.

try:
    import builtins as __builtin__
except ImportError:
    import __builtin__

def _swig_setattr_nondynamic(self, class_type, name, value, static=1):
    if (name == "thisown"):
        return self.this.own(value)
    if (name == "this"):
        if type(value).__name__ == 'SwigPyObject':
            self.__dict__[name] = value
            return
    method = class_type.__swig_setmethods__.get(name, None)
    if method:
        return method(self, value)
    if (not static):
        if _newclass:
            object.__setattr__(self, name, value)
        else:
            self.__dict__[name] = value
    else:
        raise AttributeError("You cannot add attributes to %s" % self)


def _swig_setattr(self, class_type, name, value):
    return _swig_setattr_nondynamic(self, class_type, name, value, 0)


def _swig_getattr(self, class_type, name):
    if (name == "thisown"):
        return self.this.own()
    method = class_type.__swig_getmethods__.get(name, None)
    if method:
        return method(self)
    raise AttributeError("'%s' object has no attribute '%s'" % (class_type.__name__, name))


def _swig_repr(self):
    try:
        strthis = "proxy of " + self.this.__repr__()
    except __builtin__.Exception:
        strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)

try:
    _object = object
    _newclass = 1
except __builtin__.Exception:
    class _object:
        pass
    _newclass = 0


from numpy import ndarray

def is_array(obj):
    return type(obj) is ndarray

def is_correct_type(obj, numeric_type):
    return is_array(obj) and (obj.dtype == numeric_type)

def numpy_check(obj, typecode,
                exact_size = [],
                must_be_contiguous = 1,
                allow_coersion = 0):

    if is_correct_type(obj, typecode):
        ary = obj
    elif allow_coersion:
        ary = asarray(obj,typecode)
    else:
        raise TypeError("input is not an array or the array has the wrong type")

    if must_be_contiguous and not ary.flags["CONTIGUOUS"]:
        if allow_coersion:
            ary = ary.copy()
        else:
            raise TypeError("input array must be contiguous")

# check number of dimensions
    required_dims = len(exact_size)
    if required_dims and required_dims != len(ary.shape):
        raise ValueError("The input array does not have the correct shape")

# check exact shape of each dimension
    cnt = 0
    for desired,actual in zip(exact_size,ary.shape):
        if desired != -1 and desired != actual:
            raise ValueError("The %d dimensions of the array has the wrong shape" % (cnt))
        cnt += 1

    return ary

pix_format_undefined = _plat_support.pix_format_undefined
pix_format_gray8 = _plat_support.pix_format_gray8
pix_format_rgb555 = _plat_support.pix_format_rgb555
pix_format_rgb565 = _plat_support.pix_format_rgb565
pix_format_rgb24 = _plat_support.pix_format_rgb24
pix_format_bgr24 = _plat_support.pix_format_bgr24
pix_format_rgba32 = _plat_support.pix_format_rgba32
pix_format_argb32 = _plat_support.pix_format_argb32
pix_format_abgr32 = _plat_support.pix_format_abgr32
pix_format_bgra32 = _plat_support.pix_format_bgra32
end_of_pix_formats = _plat_support.end_of_pix_formats
class PixelMap(_object):
    __swig_setmethods__ = {}
    __setattr__ = lambda self, name, value: _swig_setattr(self, PixelMap, name, value)
    __swig_getmethods__ = {}
    __getattr__ = lambda self, name: _swig_getattr(self, PixelMap, name)
    __repr__ = _swig_repr
    __swig_destroy__ = _plat_support.delete_PixelMap
    __del__ = lambda self: None

    def __init__(self, width, height, format, clear_val, bottom_up):
        this = _plat_support.new_PixelMap(width, height, format, clear_val, bottom_up)
        try:
            self.this.append(this)
        except __builtin__.Exception:
            self.this = this

    def convert_to_argb32string(self):
        return _plat_support.PixelMap_convert_to_argb32string(self)

    def set_bmp_array(self):
        self.bmp_array = pixel_map_as_unowned_array(self)
        return self


PixelMap_swigregister = _plat_support.PixelMap_swigregister
PixelMap_swigregister(PixelMap)


def pixel_map_as_unowned_array(pix_map):
    return _plat_support.pixel_map_as_unowned_array(pix_map)
pixel_map_as_unowned_array = _plat_support.pixel_map_as_unowned_array
# This file is compatible with both classic and new-style classes.


