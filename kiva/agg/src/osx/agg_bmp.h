// -*- c++ -*-
// (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
// All rights reserved.
//
// This software is provided without warranty under the terms of the BSD
// license included in LICENSE.txt and may be redistributed only under
// the conditions described in the aforementioned license. The license
// is also available online at http://www.enthought.com/licenses/BSD.txt
//
// Thanks for using Enthought open source!
#ifndef AGG_MAC_BMP_INCLUDED
#define AGG_MAC_BMP_INCLUDED

#include "Python.h"

#include "agg_basics.h"
#include "agg_rendering_buffer.h"
#include "util/agg_color_conv_rgb8.h"

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

    class pixel_map
    {
    public:
      pixel_map(unsigned width, unsigned height, pix_format_e format,
                unsigned clear_val, bool bottom_up);
      ~pixel_map();

      void init_platform(pix_format_e format, bool bottom_up);

      unsigned calc_row_len(unsigned width, unsigned bits_per_pixel);

      pix_format_e get_pix_format() const;
      unsigned char* buf();
      unsigned char* buf2();
      unsigned       width() const;
      unsigned       height() const;
      int            stride();
      unsigned       bpp() const { return m_bpp; }
      rendering_buffer& rbuf() { return m_rbuf_window; }
      rendering_buffer& rbuf2() { return m_rbuf_window2; }
      PyObject* convert_to_argb32string() const;

      pix_format_e m_format;
      pix_format_e m_sys_format;

    private:
      void        destroy();
      void        create(unsigned width,
                         unsigned height,
                         unsigned clear_val=256);

      unsigned char*   m_buf;
      unsigned char*   m_buf2;
      rendering_buffer m_rbuf_window;
      rendering_buffer m_rbuf_window2;
      unsigned m_bpp;
      unsigned m_sys_bpp;

//    public:
//      platform_specific*  m_specific;

    };

}

#endif
