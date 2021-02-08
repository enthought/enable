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
#ifndef AGG_GTK1_SPECIFIC_INCLUDED
#define AGG_GTK1_SPECIFIC_INCLUDED

#include <X11/Xlib.h>
#include "agg_basics.h"
#include "agg_rendering_buffer.h"

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


  class x11_display {
  public:
    x11_display();
    ~x11_display();
    bool open(const char* display_name = NULL);
    void close();
    bool put_image(Window dc, XImage* image);
    XImage* create_image(const rendering_buffer* rbuf);
    void destroy_image(XImage* ximg);

  public:
    Display*             m_display;    
    int                  m_screen;
    int                  m_depth;
    Visual*              m_visual;
    Window               m_dc;
    GC                   m_gc;
    unsigned             m_sys_bpp;

  };


  //------------------------------------------------------------------------
  class platform_specific
  {
    
    static x11_display x11;

  public:
    platform_specific(pix_format_e format, bool flip_y);
    ~platform_specific();
    void display_pmap(Window dc, const rendering_buffer* src);
    void destroy();

    static unsigned calc_row_len(unsigned width, unsigned bits_per_pixel);

  private:
    bool init();

  public:
    unsigned             m_bpp;         // init()
    bool                 m_flip_y;      // platform_specific()
    XImage*              m_ximage;      // display_pmap()
    pix_format_e         m_format;      // platform_specific()

  private:
    int                  m_byte_order;  // init()
    unsigned             m_sys_bpp;     // init()
    pix_format_e         m_sys_format;  // init()
    
  };

}

#endif
