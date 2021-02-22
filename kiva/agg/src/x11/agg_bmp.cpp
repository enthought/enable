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
#include <X11/Xutil.h>
#include "x11/agg_bmp.h"
#include "x11/agg_platform_specific.h"
/* #include <agg_pixfmt_rgba32.h> */
#include "agg_pixfmt_rgb.h"
#include "agg_pixfmt_rgba.h"
#include "agg_color_rgba.h"

#include "bytesobject.h"

#ifdef NUMPY
#include "numpy/arrayobject.h"
# ifndef PyArray_SBYTE
#  include "numpy/noprefix.h"
#  include "numpy/oldnumeric.h"
#  include "numpy/old_defines.h"
# endif
#else
#include "Numeric/arrayobject.h"
#define PyArray_UBYTELTR 'b'
#endif

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
		       unsigned clear_val, bool bottom_up):
    m_bmp(0),
    m_buf(0),
    m_specific(new platform_specific(format, bottom_up))
  {
    DEBUG_MTH5("pixel_map::pixel_map(%d,%d,%d,%d,%d)",width,height,format,clear_val,bottom_up);
    m_bpp = m_specific->m_bpp;
    create(width, height, clear_val);
  }

  //------------------------------------------------------------------------
  pixel_map::~pixel_map()
  {
    DEBUG_MTH("pixel_map::~pixel_map");
    destroy();

    delete m_specific;
  }

  //------------------------------------------------------------------------
  void pixel_map::destroy()
  {
    if (m_specific->m_ximage != 0)
    {
        m_specific->destroy();
    } else if(m_bmp)
    {
        delete [] (unsigned char*)m_bmp;
    }

    m_bmp  = 0;
    m_buf = 0;

  }

  //------------------------------------------------------------------------
  void pixel_map::create(unsigned width,
			 unsigned height,
			 unsigned clear_val)
  {
    destroy();
    if(width == 0)  width = 1;
    if(height == 0) height = 1;

    unsigned row_len = platform_specific::calc_row_len(width, m_bpp);
    unsigned img_size = row_len * height;

    m_bmp = (Pixmap*) new unsigned char[img_size];
    m_buf = (unsigned char*)m_bmp;

    if(clear_val <= 255) {
      memset(m_buf, clear_val, img_size);
    }

    m_rbuf_window.attach(m_buf, width, height,
			 (m_specific->m_flip_y ? -row_len : row_len));

  }

  //------------------------------------------------------------------------
  void pixel_map::draw(Window dc, int x, int y, double scale) const
  {
    DEBUG_MTH("pixel_map::draw");
    if(m_bmp == 0 || m_buf == 0) return;
    m_specific->display_pmap(dc, &m_rbuf_window);
  }

  pix_format_e pixel_map::get_pix_format() const {
    return m_specific->m_format;
  }

  unsigned char* pixel_map::buf() { return m_buf; }
  unsigned       pixel_map::width() const { return m_rbuf_window.width(); }
  unsigned       pixel_map::height() const { return m_rbuf_window.height(); }
  unsigned       pixel_map::stride() const { return platform_specific::calc_row_len(width(),m_bpp); }

  PyObject* pixel_map::convert_to_rgbarray() const {
    unsigned w = width();
    unsigned h = height();
    pix_format_e format = get_pix_format();
    rgba8 c;
    unsigned i,j;
    npy_intp dims[3];
    PyObject* arr = NULL;
    char* data = NULL;
    dims[0] = w;
    dims[1] = h;
    dims[2] = 3;
    import_array();
    arr = PyArray_SimpleNew(3,dims,PyArray_BYTE);
    if (arr==NULL)
      return NULL;
    data = ((PyArrayObject *)arr)->data;

    switch (format) {
    case pix_format_bgra32:
      {
	pixfmt_bgra32 r((rendering_buffer&)m_rbuf_window);

	for (j=0;j<h;++j)
	  for (i=0;i<w;++i)
	    {
	      c = r.pixel(i,h-j-1);
	      *(data++) = (char)c.r;
	      *(data++) = (char)c.g;
	      *(data++) = (char)c.b;
	    }
      }
      break;
    case pix_format_rgb24:
      {
	pixfmt_rgb24 r((rendering_buffer&)m_rbuf_window);

	for (j=0;j<h;++j)
	{
	  memcpy(data, r.row_ptr(h-j-1), w*3);
	  data += w*3;
	}
      }
      break;
    default:
      fprintf(stderr,"pix_format %d not handled!\n",format);
    }
    return arr;
  }

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

#ifdef WX_INFO
  wxImage* pixel_map::convert_to_wximage() const {
    unsigned w = width();
    unsigned h = height();
#ifdef WX_RELEASE_2_5
    wxImage* image = new wxImage(w, h, false);
#else
    wxImage* image = new wxImage(w, h);
#endif
    unsigned char* data = image->GetData();
    pix_format_e format = get_pix_format();
    rgba8 c;
    unsigned i,j;
    switch (format) {
    case pix_format_bgra32:
#ifdef WX_RELEASE_2_5
      image->SetAlpha();
      printf("image->HasAlpha()=%d\n",image->HasAlpha());
#endif
      {
	pixel_formats_rgba32<order_bgra32> r((rendering_buffer&)m_rbuf_window);

	for (j=0;j<h;++j)
	  for (i=0;i<w;++i)
	    {
	      c = r.pixel(i,h-j-1);
	      *(data++) = (unsigned char)c.r;
	      *(data++) = (unsigned char)c.g;
	      *(data++) = (unsigned char)c.b;
#ifdef WX_RELEASE_2_5
	      image->SetAlpha((int)i,(int)j,(unsigned char)c.a);
#endif
	    }
      }
      break;
    default:
      fprintf(stderr,"pix_format %d not handled!\n",format);
    }
    return image;
  }
#endif

}
