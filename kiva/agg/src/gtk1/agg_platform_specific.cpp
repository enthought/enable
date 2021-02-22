// (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
// All rights reserved.
//
// This software is provided without warranty under the terms of the BSD
// license included in LICENSE.txt and may be redistributed only under
// the conditions described in the aforementioned license. The license
// is also available online at http://www.enthought.com/licenses/BSD.txt
//
// Thanks for using Enthought open source!
#include <stdio.h>
#include <string.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <time.h>
#include "gtk1/agg_platform_specific.h"
#include "util/agg_color_conv_rgb8.h"
#include "gtk1/agg_bmp.h"

//#define DEBUG_MTH(NAME) fprintf(stderr, NAME "\n");
#define DEBUG_MTH(NAME)

typedef struct {
    short x1, x2, y1, y2;
} Box, BOX, BoxRec, *BoxPtr;

typedef struct {
    short x, y, width, height;
}RECTANGLE, RectangleRec, *RectanglePtr;
typedef struct _XRegion {
    long size;
    long numRects;
    BOX *rects;
    BOX extents;
} REGION;



namespace agg24
{

  x11_display::x11_display():
    m_screen(0), m_depth(0), m_visual(0), m_dc(0), m_gc(0), m_sys_bpp(0)
  {
    DEBUG_MTH("x11_display::x11_display");
  }

  x11_display::~x11_display() {
    DEBUG_MTH("x11_display::~x11_display");
    close();
  }

  bool x11_display::open(const char* display_name) {
    DEBUG_MTH("x11_display::open");
    if (m_display != 0) {
      fprintf(stderr, "X11 display is opened already\n");
      return false;
    }
    //printf("Opened X11 display: %s\n", display_name);
    m_display = XOpenDisplay(display_name);
    if (m_display == 0) {
      fprintf(stderr, "Unable to open DISPLAY=%s!\n",display_name);
      return false;
    }
    m_screen = DefaultScreen(m_display);
    m_depth  = DefaultDepth(m_display, m_screen);
    m_visual = DefaultVisual(m_display, m_screen);

    switch(m_depth) {
    case 15:
    case 16:
      m_sys_bpp = 16;
      break;
    case 24:
    case 32:
      m_sys_bpp = 32;
      break;
    default:
      fprintf(stderr, "Unexpected X11 display depth=%d!\n",m_depth);
    }
    return true;
  }

  void x11_display::close() {
    DEBUG_MTH("x11_display::close");
    if (m_display != 0) {
      if (m_gc != 0)
	XFreeGC(m_display, m_gc);
      XCloseDisplay(m_display);
    }
    m_display = 0;
    m_screen = 0;
    m_depth = 0;
    m_visual = 0;
    m_dc = 0;
    m_gc = 0;
  }

  bool x11_display::put_image(Window dc, XImage* image) {
    DEBUG_MTH("x11_display::put_image");
    if (m_dc != dc) {
      if (m_gc)
	XFreeGC(m_display, m_gc);
      m_dc = dc;
      //m_gc = DefaultGC(m_display, m_screen);
      m_gc = XCreateGC(m_display, m_dc, 0, 0);
    }
    XPutImage(m_display,
	      dc,
	      m_gc,
	      image,
	      0, 0, 0, 0,
	      image->width,
	      image->height
	      );
    return true;
  }

  XImage* x11_display::create_image(const rendering_buffer* rbuf) {
    DEBUG_MTH("x11_display::create_image");
    unsigned width = rbuf->width(); 
    unsigned height = rbuf->height(); 
    return XCreateImage(m_display, 
			m_visual, //CopyFromParent, 
			m_depth, 
			ZPixmap, 
			0,
			(char*)(rbuf->buf()), 
			width,
			height, 
			m_sys_bpp,
			width * (m_sys_bpp / 8));
  }

  void x11_display::destroy_image(XImage* ximg) {
    DEBUG_MTH("x11_display::destroy_image");
    if (ximg != 0)
      XDestroyImage(ximg);
    /*
      XDestroyImage function frees both the image structure
      and the data pointed to by the image structure.
    */
  }

  //------------------------------------------------------------------------
  platform_specific::platform_specific(pix_format_e format, bool flip_y) :
    m_format(format),
    m_flip_y(flip_y),
    m_bpp(0),
    m_sys_bpp(0),
    m_sys_format(pix_format_undefined),
    m_ximage(0)
  {
    DEBUG_MTH("platform_specific::platform_specific");
    init();

  }



  platform_specific::~platform_specific() {
    DEBUG_MTH("platform_specific::~platform_specific");
  }

  x11_display platform_specific::x11;

  bool platform_specific::init()
  {
    DEBUG_MTH("platform_specific::init");
    if (x11.m_display == 0 && !x11.open()) {
      fprintf(stderr, "No X11 display available!\n");
      return false;
    }
    unsigned long r_mask = x11.m_visual->red_mask;
    unsigned long g_mask = x11.m_visual->green_mask;
    unsigned long b_mask = x11.m_visual->blue_mask;

    if(x11.m_depth < 15 || r_mask == 0 || g_mask == 0 || b_mask == 0)
      {
	fprintf(stderr,
		"There's no Visual compatible with minimal AGG requirements:\n"
		"At least 15-bit color depth and True- or DirectColor class.\n\n"
		);
	return false;
      }

    switch(m_format)
      {
      case pix_format_gray8:
	m_bpp = 8;
	break;
	
      case pix_format_rgb565:
      case pix_format_rgb555:
	m_bpp = 16;
	break;

      case pix_format_rgb24:
      case pix_format_bgr24:
	m_bpp = 24;
	break;

      case pix_format_bgra32:
      case pix_format_abgr32:
      case pix_format_argb32:
      case pix_format_rgba32:
	m_bpp = 32;
	break;
      case pix_format_undefined:
      case end_of_pix_formats:
	;
      }

    //
    // Calculate m_sys_format, m_byte_order, m_sys_bpp:
    //
    int t = 1;
    int hw_byte_order = LSBFirst;
    if(*(char*)&t == 0) hw_byte_order = MSBFirst;

    switch(x11.m_depth) {
    case 15:
      m_sys_bpp = 16;
      if(r_mask == 0x7C00 && g_mask == 0x3E0 && b_mask == 0x1F) {
	m_sys_format = pix_format_rgb555;
	m_byte_order = hw_byte_order;
      }
      break;
    case 16:
      m_sys_bpp = 16;
      if(r_mask == 0xF800 && g_mask == 0x7E0 && b_mask == 0x1F) {
	m_sys_format = pix_format_rgb565;
	m_byte_order = hw_byte_order;
      }
      break;
    case 24:
    case 32:
      m_sys_bpp = 32;
      if(g_mask == 0xFF00)
	{
	  if(r_mask == 0xFF && b_mask == 0xFF0000)
	    {
	      switch(m_format)
		{
		case pix_format_rgba32:
		  m_sys_format = pix_format_rgba32;
		  m_byte_order = LSBFirst;
		  break;
		  
		case pix_format_abgr32:
		  m_sys_format = pix_format_abgr32;
		  m_byte_order = MSBFirst;
		  break;
		  
		default:                            
		  m_byte_order = hw_byte_order;
		  m_sys_format = 
		    (hw_byte_order == LSBFirst) ?
		    pix_format_rgba32 :
		    pix_format_abgr32;
		  break;
		}
	    }
	  if(r_mask == 0xFF0000 && b_mask == 0xFF)
	    {
	      switch(m_format)
		{
		case pix_format_argb32:
		  m_sys_format = pix_format_argb32;
		  m_byte_order = MSBFirst;
		  break;
		  
		case pix_format_bgra32:
		  m_sys_format = pix_format_bgra32;
		  m_byte_order = LSBFirst;
		  break;
		  
		default:                            
		  m_byte_order = hw_byte_order;
		  m_sys_format = 
		    (hw_byte_order == MSBFirst) ?
		    pix_format_argb32 :
		    pix_format_bgra32;
		  break;
		}
	    }
	}
      break;
    }
    if(m_sys_format == pix_format_undefined)
      {
	fprintf(stderr,
		"RGB masks are not compatible with AGG pixel formats:\n"
		"R=%08x, G=%08x, B=%08x\n", r_mask, g_mask, b_mask);
	return false;
      }
    return true;
  }

#define UNHANDLED_PIX_FORMATS \
  case end_of_pix_formats: ; \
  case pix_format_gray8: ; \
  case pix_format_undefined: ;

  void platform_specific::destroy() {
    DEBUG_MTH("platform_specific::destroy");
    if (m_ximage != 0) {
      x11.destroy_image(m_ximage);
      m_ximage = 0;
    }
  }

  void platform_specific::display_pmap(Window dc, const rendering_buffer* rbuf) {
    DEBUG_MTH("platform_specific::display_map");
    if (m_format == m_sys_format) {
      if (m_ximage == 0) {
	m_ximage = x11.create_image(rbuf);
	m_ximage->byte_order = m_byte_order;
      }
      x11.put_image(dc, m_ximage);
      return;
    }
    // Optimization hint: make pmap_tmp as a private class member and reused it when possible.
    pixel_map pmap_tmp(rbuf->width(),
		       rbuf->height(),
		       m_sys_format,
		       256,
		       m_flip_y);
    rendering_buffer* rbuf2 = &pmap_tmp.rbuf();

    switch(m_sys_format)
      {
      case pix_format_rgb555:
	switch(m_format)
	  {
	  case pix_format_rgb555: color_conv(rbuf2, rbuf, color_conv_rgb555_to_rgb555()); break;
	  case pix_format_rgb565: color_conv(rbuf2, rbuf, color_conv_rgb565_to_rgb555()); break;
	  case pix_format_rgb24:  color_conv(rbuf2, rbuf, color_conv_rgb24_to_rgb555());  break;
	  case pix_format_bgr24:  color_conv(rbuf2, rbuf, color_conv_bgr24_to_rgb555());  break;
	  case pix_format_rgba32: color_conv(rbuf2, rbuf, color_conv_rgba32_to_rgb555()); break;
	  case pix_format_argb32: color_conv(rbuf2, rbuf, color_conv_argb32_to_rgb555()); break;
	  case pix_format_bgra32: color_conv(rbuf2, rbuf, color_conv_bgra32_to_rgb555()); break;
	  case pix_format_abgr32: color_conv(rbuf2, rbuf, color_conv_abgr32_to_rgb555()); break;
	    UNHANDLED_PIX_FORMATS;
	  }
	break;
                    
      case pix_format_rgb565:
	switch(m_format)
	  {
	  case pix_format_rgb555: color_conv(rbuf2, rbuf, color_conv_rgb555_to_rgb565()); break;
	  case pix_format_rgb565: color_conv(rbuf2, rbuf, color_conv_rgb565_to_rgb565()); break;
	  case pix_format_rgb24:  color_conv(rbuf2, rbuf, color_conv_rgb24_to_rgb565());  break;
	  case pix_format_bgr24:  color_conv(rbuf2, rbuf, color_conv_bgr24_to_rgb565());  break;
	  case pix_format_rgba32: color_conv(rbuf2, rbuf, color_conv_rgba32_to_rgb565()); break;
	  case pix_format_argb32: color_conv(rbuf2, rbuf, color_conv_argb32_to_rgb565()); break;
	  case pix_format_bgra32: color_conv(rbuf2, rbuf, color_conv_bgra32_to_rgb565()); break;
	  case pix_format_abgr32: color_conv(rbuf2, rbuf, color_conv_abgr32_to_rgb565()); break;
	    UNHANDLED_PIX_FORMATS;
	  }
	break;
	
      case pix_format_rgba32:
	switch(m_format)
	  {
	  case pix_format_rgb555: color_conv(rbuf2, rbuf, color_conv_rgb555_to_rgba32()); break;
	  case pix_format_rgb565: color_conv(rbuf2, rbuf, color_conv_rgb565_to_rgba32()); break;
	  case pix_format_rgb24:  color_conv(rbuf2, rbuf, color_conv_rgb24_to_rgba32());  break;
	  case pix_format_bgr24:  color_conv(rbuf2, rbuf, color_conv_bgr24_to_rgba32());  break;
	  case pix_format_rgba32: color_conv(rbuf2, rbuf, color_conv_rgba32_to_rgba32()); break;
	  case pix_format_argb32: color_conv(rbuf2, rbuf, color_conv_argb32_to_rgba32()); break;
	  case pix_format_bgra32: color_conv(rbuf2, rbuf, color_conv_bgra32_to_rgba32()); break;
	  case pix_format_abgr32: color_conv(rbuf2, rbuf, color_conv_abgr32_to_rgba32()); break;
	    UNHANDLED_PIX_FORMATS;
	  }
	break;
	
      case pix_format_abgr32:
	switch(m_format)
	  {
	  case pix_format_rgb555: color_conv(rbuf2, rbuf, color_conv_rgb555_to_abgr32()); break;
	  case pix_format_rgb565: color_conv(rbuf2, rbuf, color_conv_rgb565_to_abgr32()); break;
	  case pix_format_rgb24:  color_conv(rbuf2, rbuf, color_conv_rgb24_to_abgr32());  break;
	  case pix_format_bgr24:  color_conv(rbuf2, rbuf, color_conv_bgr24_to_abgr32());  break;
	  case pix_format_abgr32: color_conv(rbuf2, rbuf, color_conv_abgr32_to_abgr32()); break;
	  case pix_format_rgba32: color_conv(rbuf2, rbuf, color_conv_rgba32_to_abgr32()); break;
	  case pix_format_argb32: color_conv(rbuf2, rbuf, color_conv_argb32_to_abgr32()); break;
	  case pix_format_bgra32: color_conv(rbuf2, rbuf, color_conv_bgra32_to_abgr32()); break;
	    UNHANDLED_PIX_FORMATS;
	  }
	break;
	
      case pix_format_argb32:
	switch(m_format)
	  {
	  case pix_format_rgb555: color_conv(rbuf2, rbuf, color_conv_rgb555_to_argb32()); break;
	  case pix_format_rgb565: color_conv(rbuf2, rbuf, color_conv_rgb565_to_argb32()); break;
	  case pix_format_rgb24:  color_conv(rbuf2, rbuf, color_conv_rgb24_to_argb32());  break;
	  case pix_format_bgr24:  color_conv(rbuf2, rbuf, color_conv_bgr24_to_argb32());  break;
	  case pix_format_rgba32: color_conv(rbuf2, rbuf, color_conv_rgba32_to_argb32()); break;
	  case pix_format_argb32: color_conv(rbuf2, rbuf, color_conv_argb32_to_argb32()); break;
	  case pix_format_abgr32: color_conv(rbuf2, rbuf, color_conv_abgr32_to_argb32()); break;
	  case pix_format_bgra32: color_conv(rbuf2, rbuf, color_conv_bgra32_to_argb32()); break;
	    UNHANDLED_PIX_FORMATS;
	  }
	break;
	
      case pix_format_bgra32:
	switch(m_format)
	  {
	  case pix_format_rgb555: color_conv(rbuf2, rbuf, color_conv_rgb555_to_bgra32()); break;
	  case pix_format_rgb565: color_conv(rbuf2, rbuf, color_conv_rgb565_to_bgra32()); break;
	  case pix_format_rgb24:  color_conv(rbuf2, rbuf, color_conv_rgb24_to_bgra32());  break;
	  case pix_format_bgr24:  color_conv(rbuf2, rbuf, color_conv_bgr24_to_bgra32());  break;
	  case pix_format_rgba32: color_conv(rbuf2, rbuf, color_conv_rgba32_to_bgra32()); break;
	  case pix_format_argb32: color_conv(rbuf2, rbuf, color_conv_argb32_to_bgra32()); break;
	  case pix_format_abgr32: color_conv(rbuf2, rbuf, color_conv_abgr32_to_bgra32()); break;
	  case pix_format_bgra32: color_conv(rbuf2, rbuf, color_conv_bgra32_to_bgra32()); break;
	    UNHANDLED_PIX_FORMATS;
	  }
	break;
	UNHANDLED_PIX_FORMATS;
      }
    pmap_tmp.draw(dc);

  }

  unsigned platform_specific::calc_row_len(unsigned width, unsigned bits_per_pixel) {
    return width * (bits_per_pixel / 8);
  }

}
