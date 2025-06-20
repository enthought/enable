// -*- c++ -*-
// (C) Copyright 2005-2022 Enthought, Inc., Austin, TX
// All rights reserved.
//
// This software is provided without warranty under the terms of the BSD
// license included in LICENSE.txt and may be redistributed only under
// the conditions described in the aforementioned license. The license
// is also available online at http://www.enthought.com/licenses/BSD.txt
//
// Thanks for using Enthought open source!

%module plat_support

%include numeric.i

%{

#include "osx/agg_bmp.h"
namespace agg24
{
    PyObject* pixel_map_as_unowned_array(agg24::pixel_map& pix_map)
    {
        npy_intp dims[3];
        npy_intp rows = pix_map.height();
        npy_intp cols = pix_map.width();
        npy_intp depth = pix_map.bpp() / 8;

        dims[0] = rows;
        dims[1] = cols;
        dims[2] = depth;

        return PyArray_SimpleNewFromData(3,dims,NPY_UINT8,(void*)pix_map.buf());
    }
}

%}

// More permissive unsigned typemap that converts any numeric type to an
// unsigned value.  It is cleared at the end of this file.
%typemap(in) unsigned
{
    PyObject* obj = PyNumber_Long($input);
    if (PyErr_Occurred()) SWIG_fail;
    $1 = (unsigned) PyLong_AsLong(obj);
    if (PyErr_Occurred()) SWIG_fail;
}

namespace agg24
{
    enum pix_format_e
    {
        pix_format_undefined = 0,  // By default. No conversions are applied
        pix_format_gray8,          // Simple 256 level grayscale
        pix_format_rgb555,         // 15 bit rgb. Depends on the byte ordering!
        pix_format_rgb565,         // 16 bit rgb. Depends on the byte ordering!
        pix_format_rgb24,          // R-G-B, one byte per color component
        pix_format_bgr24,          // B-G-R, native win32 BMP format.
        pix_format_rgba32,         // R-G-B-A, one byte per color component
        pix_format_argb32,         // A-R-G-B, native MAC format
        pix_format_abgr32,         // A-B-G-R, one byte per color component
        pix_format_bgra32,         // B-G-R-A, native win32 BMP format

        end_of_pix_formats
    };

    %rename(PixelMap) pixel_map;
    class pixel_map
    {
    public:
        ~pixel_map();
        pixel_map(unsigned width, unsigned height, pix_format_e format,
                  unsigned clear_val, bool bottom_up);
    public:
      PyObject* convert_to_argb32string() const;

      %pythoncode
      %{
        def set_bmp_array(self):
            self.bmp_array = pixel_map_as_unowned_array(self)
            return self

      %}
    };

PyObject* pixel_map_as_unowned_array(pixel_map& pix_map);
}

// clear the "permissive" unsigned typemap we are using.
%typemap(in) unsigned;
