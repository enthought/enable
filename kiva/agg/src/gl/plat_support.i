// -*- c++ -*-
// OpenGL support for AGG
// Author: Robert Kern
// I've consolidated agg_platform_specific and agg_bmp in the process of
// understanding the code.

// plat_support defines a function resize_gl(width, height) which should be called
// every time the window gets resized. All OpenGL initialization and glViewport
// calls need to be done by the widget toolkit.

// Currently, OpenGL support is only tested with wxWidgets 2.5.1.5 on MacOS X
// version 10.3

%module plat_support

%include numeric.i

%{

#include "gl/agg_bmp.h"
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

    void resize_gl(unsigned width, unsigned height)
    {
        GLint viewport[4];

        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        gluOrtho2D(0.0, (GLfloat)width, 0.0, (GLfloat)height);
        glPixelZoom(1.0, -1.0);
        glGetIntegerv(GL_VIEWPORT, viewport);
        glRasterPos2d(0.0, ((double)height*height)/viewport[3]);
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

    %name(PixelMap) class pixel_map
    {
    public:
        ~pixel_map();
        pixel_map(unsigned width, unsigned height, pix_format_e format,
                  unsigned clear_val, bool bottom_up);
    public:
       %feature("shadow") draw(int x, int y, double scale)
       %{
       def draw(self, x=0, y=0, scale=1.0):
           # fix me: brittle becuase we are hard coding 
           # module and class name.  Done cause SWIG 1.3.24 does
           # some funky overloading stuff in it that breaks keyword
           # arguments.
           result = _plat_support.PixelMap_draw(self, x, y, scale)
           return result
       %}
      void draw(int x, int y, double scale);
      PyObject* convert_to_argb32string() const;

      %pythoncode
      %{

    def set_bmp_array(self):
         self.bmp_array = pixel_map_as_unowned_array(self)
         return self

    def draw_to_glcanvas(self, x, y):
        self.draw(x, y)

      %}
    };

PyObject* pixel_map_as_unowned_array(pixel_map& pix_map);
void resize_gl(unsigned width, unsigned height);
}

// clear the "permissive" unsigned typemap we are using.
%typemap(in) unsigned;
