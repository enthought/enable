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
#include "win32/agg_bmp.h"
#include "win32/agg_platform_specific.h"

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
                         unsigned clear_val, bool bottom_up):
                            m_buf(NULL),
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
        DEBUG_MTH("pixel_map::destroy()");
        if (m_specific->m_bimage != NULL)
        {
            DEBUG_MTH("pixel_map::destroy() m_bimage != NULL");
            m_specific->destroy();
            m_buf = NULL;
        }

        if (m_buf != NULL)
        {
            delete[] m_buf;
            m_buf = NULL;
        }
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

        m_buf = new unsigned char[img_size];

        if(clear_val <= 255) {
            memset(m_buf, clear_val, img_size);
        }

        m_rbuf_window.attach(m_buf, width, height,
            (m_specific->m_flip_y ? -row_len : row_len));

    }

    //------------------------------------------------------------------------
    void pixel_map::draw(HDC dc, int draw_x, int draw_y, int draw_width,
    		             int draw_height) const
    {
        DEBUG_MTH("pixel_map::draw");
        if (m_buf != NULL)
        {
            m_specific->display_pmap(dc, &m_rbuf_window, draw_x, draw_y,
            		                 draw_width, draw_height);
        }
    }

    pix_format_e pixel_map::get_pix_format() const {
        return m_specific->m_format;
    }

    unsigned char* pixel_map::buf() { return m_buf; }
    unsigned       pixel_map::width() const { return m_rbuf_window.width(); }
    unsigned       pixel_map::height() const { return m_rbuf_window.height(); }
    int            pixel_map::stride() const { return platform_specific::calc_row_len(width(), m_bpp); }

    // Convert to a Python string containing 32 bit ARGB values.
    PyObject* pixel_map::convert_to_argb32string() const
    {
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
