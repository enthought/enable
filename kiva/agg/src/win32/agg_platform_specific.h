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
#ifndef AGG_WIN32_SPECIFIC_INCLUDED
#define AGG_WIN32_SPECIFIC_INCLUDED

#include <windows.h>
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
    
    typedef struct {
        BITMAPINFO* bmp;
        unsigned char* data;
    } BImage;
    
    class dib_display {
        
        public:
            dib_display();
            ~dib_display();
            bool put_image(HDC dc, BImage* image, int draw_x=-1, int draw_y=-1,
            		       int draw_width=-1, int draw_height=-1);
            BImage* create_image(const rendering_buffer* rbuf, unsigned bits_per_pixel);
            void destroy_image(BImage* image);
        
        private:
            static unsigned calc_header_size(BITMAPINFO *bmp);
            static unsigned calc_palette_size(BITMAPINFO *bmp);
            static unsigned calc_palette_size(unsigned  clr_used, unsigned bits_per_pixel);
    };
    


    class platform_specific
    {
        
        static dib_display dib;
        
        public:
            platform_specific(pix_format_e format, bool flip_y);
            ~platform_specific() {}
            void display_pmap(HDC dc, const rendering_buffer* src,
            		          int draw_x=-1, int draw_y=-1,
            		          int draw_width=-1, int draw_height=-1);
            void destroy();
        
            static unsigned calc_row_len(unsigned width, unsigned bits_per_pixel);
        
        
            unsigned             m_bpp;
            bool                 m_flip_y;
            BImage*              m_bimage;
            pix_format_e  m_format;
        
        private:
            pix_format_e  m_sys_format;
            unsigned      m_sys_bpp;
        
    };

}

#endif
