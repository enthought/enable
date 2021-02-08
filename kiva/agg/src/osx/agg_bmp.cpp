// (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
// All rights reserved.
//
// This software is provided without warranty under the terms of the BSD
// license included in LICENSE.txt and may be redistributed only under
// the conditions described in the aforementioned license. The license
// is also available online at http://www.enthought.com/licenses/BSD.txt
//
// Thanks for using Enthought open source!
#include <string.h>
#include <stdio.h>

#include "osx/agg_bmp.h"

#include "agg_pixfmt_rgba.h"
#include "agg_color_rgba.h"

#include "bytesobject.h"

#if 0
#define DEBUG_MTH(NAME) fprintf(stderr, NAME "\n");
#define DEBUG_MTH2(STR,ARG1,ARG2) fprintf(stderr, STR "\n",(ARG1),(ARG2));
#define DEBUG_MTH5(STR,ARG1,ARG2,ARG3,ARG4,ARG5) fprintf(stderr, STR "\n",(ARG1),(ARG2),(ARG3),(ARG4),(ARG5));
#else
#define DEBUG_MTH(NAME)
#define DEBUG_MTH2(STR,ARG1,ARG2)
#define DEBUG_MTH5(STR,ARG1,ARG2,ARG3,ARG4,ARG5)
#endif


namespace agg24
{

  //------------------------------------------------------------------------
  pixel_map::pixel_map(unsigned width, unsigned height, pix_format_e format,
                       unsigned clear_val, bool bottom_up)
  : m_format(format)
  , m_buf(NULL)
  ,  m_buf2(NULL)
//,  m_specific(new platform_specific(format, bottom_up))
  {
    DEBUG_MTH5("pixel_map::pixel_map(%d,%d,%d,%d,%d)",width,height,format,clear_val,bottom_up);

    init_platform(format, bottom_up);
    create(width, height, clear_val);

  }



  //------------------------------------------------------------------------
  void pixel_map::init_platform(pix_format_e format, bool bottom_up)
  {
    switch(m_format)
    {
    case pix_format_gray8:
      m_sys_format = pix_format_gray8;
      m_bpp = 8;
      m_sys_bpp = 8;
      break;

    case pix_format_rgb555:
    case pix_format_rgb565:
      m_sys_format = pix_format_rgb565;
      m_bpp = 16;
      m_sys_bpp = 16;
      break;

    case pix_format_rgb24:
      m_sys_format = pix_format_rgb24;
      m_bpp = 24;
      m_sys_bpp = 24;
      break;

    case pix_format_bgr24:
      m_sys_format = pix_format_bgr24;
      m_bpp = 24;
      m_sys_bpp = 24;
      break;

    case pix_format_bgra32:
    case pix_format_abgr32:
      m_sys_format = pix_format_bgra32;
      m_bpp = 32;
      m_sys_bpp = 32;
      break;

    case pix_format_argb32:
    case pix_format_rgba32:
      m_sys_format = pix_format_rgba32;
      m_bpp = 32;
      m_sys_bpp = 32;
      break;

    case pix_format_undefined:
    case end_of_pix_formats:
      ;
    }

  }

  //------------------------------------------------------------------------
  unsigned pixel_map::calc_row_len(unsigned width, unsigned bits_per_pixel)
  {
    unsigned n = width;
    unsigned k;
    switch(bits_per_pixel)
      {
      case  1: k = n;
        n = n >> 3;
        if(k & 7) n++;
        break;
      case  4: k = n;
        n = n >> 1;
        if(k & 3) n++;
        break;
      case  8:
        break;
      case 16: n = n << 1;
        break;
      case 24: n = (n << 1) + n;
        break;
      case 32: n = n << 2;
        break;
      default: n = 0;
        break;
      }
    return ((n + 3) >> 2) << 2;
  }

  //------------------------------------------------------------------------
  pixel_map::~pixel_map()
  {
    DEBUG_MTH("pixel_map::~pixel_map");
    destroy();
  }

  //------------------------------------------------------------------------
  void pixel_map::destroy()
  {
    if (m_buf) {
        delete [] (unsigned char*)m_buf;
    }
    m_buf = NULL;
    if (m_buf2) {
        delete [] (unsigned char*)m_buf2;
    }
    m_buf2 = NULL;
  }

  //------------------------------------------------------------------------
  void pixel_map::create(unsigned width,
             unsigned height,
             unsigned clear_val)
  {
    destroy();
    if(width == 0)  width = 1;
    if(height == 0) height = 1;

    unsigned row_len = calc_row_len(width, m_bpp);
    unsigned img_size = row_len * height;

    m_buf = new unsigned char[img_size];

    if(clear_val <= 255) {
      memset(m_buf, clear_val, img_size);
    }

    m_rbuf_window.attach(m_buf, width, height, row_len);

    if (m_format != m_sys_format)
    {
        row_len = calc_row_len(width, m_sys_bpp);
        img_size = row_len*height;
        m_buf2 = new unsigned char[img_size];
        if (clear_val <= 255) {
            memset(m_buf2, clear_val, img_size);
        }
        m_rbuf_window2.attach(m_buf2, width, height, row_len);
    }

  }


  pix_format_e pixel_map::get_pix_format() const {
      return m_format;
  }


  unsigned char* pixel_map::buf() { return m_buf; }
  unsigned char* pixel_map::buf2() { return m_buf2; }
  unsigned       pixel_map::width() const { return m_rbuf_window.width(); }
  unsigned       pixel_map::height() const { return m_rbuf_window.height(); }
  int            pixel_map::stride() { return calc_row_len(width(), m_bpp); }

  // Convert to a Python string containing 32 bit ARGB values.
  PyObject* pixel_map::convert_to_argb32string() const {
    unsigned w = width();
    unsigned h = height();

    PyObject *str = PyBytes_FromStringAndSize(NULL, w * h * 4);

    if (str == NULL)
      return NULL;

    unsigned *data = (unsigned *)PyBytes_AS_STRING(str);

    pix_format_e format = get_pix_format();

    switch (format)
    {
    case pix_format_bgra32:
      {
        pixfmt_bgra32 r((rendering_buffer &)m_rbuf_window);

        for (unsigned j = 0; j < h; ++j)
          for (unsigned i = 0; i < w; ++i)
          {
            rgba8 c = r.pixel(i, h - j - 1);

            *data++ = (((unsigned char)c.a) << 24) |
                      (((unsigned char)c.r) << 16) |
                      (((unsigned char)c.g) << 8) |
                      ((unsigned char)c.b);
          }
      }
      break;

    default:
      Py_DECREF(str);
      PyErr_Format(PyExc_ValueError, "pix_format %d not handled", format);
      return NULL;
    }

    return str;
  }
}
