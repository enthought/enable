// (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
// All rights reserved.
//
// This software is provided without warranty under the terms of the BSD
// license included in LICENSE.txt and may be redistributed only under
// the conditions described in the aforementioned license. The license
// is also available online at http://www.enthought.com/licenses/BSD.txt
//
// Thanks for using Enthought open source!

#include <windows.h>
#include <string.h>
#include <stdio.h>
#include "agg_basics.h"
#include "util/agg_color_conv_rgb8.h"
#include "win32/agg_platform_specific.h"
#include "win32/agg_bmp.h"

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

    dib_display::dib_display() {
    }

    dib_display::~dib_display() {
    }

    BImage* dib_display::create_image(const rendering_buffer* rbuf,
                                    unsigned bits_per_pixel)

    {
        DEBUG_MTH("dib_display::create_image");
        unsigned width = rbuf->width(); 
        unsigned height = rbuf->height();
        unsigned line_len = platform_specific::calc_row_len(width, bits_per_pixel);
        unsigned img_size = line_len * height;
        unsigned rgb_size = calc_palette_size(0, bits_per_pixel) * sizeof(RGBQUAD);
        unsigned full_size = sizeof(BITMAPINFOHEADER) + rgb_size;// + img_size;

        BITMAPINFO *bmp = (BITMAPINFO *) new unsigned char[full_size];

        bmp->bmiHeader.biSize   = sizeof(BITMAPINFOHEADER);
        bmp->bmiHeader.biWidth  = width;
        bmp->bmiHeader.biHeight = -height;
        bmp->bmiHeader.biPlanes = 1;
        bmp->bmiHeader.biBitCount = (unsigned short)bits_per_pixel;
        bmp->bmiHeader.biCompression = 0;
        bmp->bmiHeader.biSizeImage = img_size;
        bmp->bmiHeader.biXPelsPerMeter = 0;
        bmp->bmiHeader.biYPelsPerMeter = 0;
        bmp->bmiHeader.biClrUsed = 0;
        bmp->bmiHeader.biClrImportant = 0;

        RGBQUAD *rgb = (RGBQUAD*)(((unsigned char*)bmp) + sizeof(BITMAPINFOHEADER));
        unsigned brightness;
        unsigned i;
        for(i = 0; i < rgb_size; i++)
        {
            brightness = (255 * i) / (rgb_size - 1);
            rgb->rgbBlue =
            rgb->rgbGreen =  
            rgb->rgbRed = (unsigned char)brightness; 
            rgb->rgbReserved = 0;
            rgb++;
        }

        BImage* image = new BImage;
        image->bmp = bmp;
        image->data = (unsigned char*)rbuf->buf();

        return image;

    }

    void dib_display::destroy_image(BImage* image)

    {
        if (image != NULL)

        {
            delete [] (unsigned char*)(image->bmp);
            delete [] (unsigned char*)(image->data);  // using XDestroyImage behavior.
            delete image;
        }
    }

    unsigned dib_display::calc_header_size(BITMAPINFO *bmp)
    {
        if (bmp == NULL)

        {

            return 0;

        } else {

            return sizeof(BITMAPINFOHEADER) + sizeof(RGBQUAD) * calc_palette_size(bmp);

        }
    }

    unsigned dib_display::calc_palette_size(BITMAPINFO *bmp)
    {
        if (bmp == 0)

        {

            return 0;

        } else {

            return calc_palette_size(bmp->bmiHeader.biClrUsed, bmp->bmiHeader.biBitCount);

        }
    }
    
    unsigned dib_display::calc_palette_size(unsigned  clr_used, unsigned bits_per_pixel)
    {
        int palette_size = 0;
        
        if(bits_per_pixel <= 8)
        {
            palette_size = clr_used;
            if(palette_size == 0)
            {
                palette_size = 1 << bits_per_pixel;
            }
        }
        return palette_size;
    }


    bool dib_display::put_image(HDC dc, BImage *image,
    		                    int draw_x, int draw_y,
    		                    int draw_width, int draw_height)

    {
        DEBUG_MTH("dib_display::put_image");
        // If specified, only do a partial blit of the image
        int dest_x, dest_y, src_x, src_y, src_width, src_height;

        
        if (draw_x != -1 &&
            draw_y != -1 &&
            draw_width != -1 &&
            draw_height != -1) {
        	src_x = draw_x;
        	src_y = draw_y;
        	dest_y = -image->bmp->bmiHeader.biHeight - draw_y - draw_height;
        	dest_x = draw_x;
        	src_width = draw_width;
        	src_height = draw_height;
        }
        else {
        	src_x = 0;
        	src_y = 0;
        	dest_x = 0;
        	dest_y = 0;
	        src_width = image->bmp->bmiHeader.biWidth;
	        src_height = -image->bmp->bmiHeader.biHeight;
        }
	        
        ::SetDIBitsToDevice(
	        dc,                           // handle to device context
	        dest_x,                       // x-coordinate of upper-left corner of dest 
	        dest_y,                       // y-coordinate of upper-left corner of  dest
	        src_width,                    // source rectangle width
	        src_height,                  // source rectangle height
	        src_x,                        // x-coordinate of lower-left corner of source
	        src_y,                        // y-coordinate of lower-left corner of source
	        0,                            // first scan line in array
	        -image->bmp->bmiHeader.biHeight,      // number of scan lines
	        image->data,                  // address of array with DIB bits
	        image->bmp,                   // address of structure with bitmap info.
	        DIB_RGB_COLORS                // RGB or palette indexes
	        );
        return true;
    }


    dib_display platform_specific::dib;
    
    //------------------------------------------------------------------------
    platform_specific::platform_specific(pix_format_e format, bool flip_y) :
                                            m_bpp(0),
                                            m_flip_y(flip_y),
                                            m_bimage(0),
                                            m_format(format),
                                            m_sys_format(pix_format_undefined),
                                            m_sys_bpp(0)
    {  
        switch(m_format)
        {
        case pix_format_gray8:
            m_sys_format = pix_format_gray8;
            m_bpp = 8;
            m_sys_bpp = 8;
            break;
            
        case pix_format_rgb565:
        case pix_format_rgb555:
            m_sys_format = pix_format_rgb555;
            m_bpp = 16;
            m_sys_bpp = 16;
            break;
            
        case pix_format_rgb24:
        case pix_format_bgr24:
            m_sys_format = pix_format_bgr24;
            m_bpp = 24;
            m_sys_bpp = 24;
            break;
        case pix_format_bgra32:
        case pix_format_abgr32:
        case pix_format_argb32:
        case pix_format_rgba32:
            m_sys_format = pix_format_bgra32;
            m_bpp = 32;
            m_sys_bpp = 32;
            break;
        case pix_format_undefined:
        case end_of_pix_formats:
            ;
        }
    }

    void platform_specific::destroy()

    {
        DEBUG_MTH("platform_specific::destroy");
        if (m_bimage != NULL)

        {
            dib.destroy_image(m_bimage);
            m_bimage = 0;
        }
    }


    //------------------------------------------------------------------------
    void platform_specific::display_pmap(HDC dc, const rendering_buffer* rbuf,
    									 int draw_x, int draw_y,
    									 int draw_width, int draw_height)
    {
        if(m_sys_format == m_format)
        {
            if (m_bimage == 0)

            {
                m_bimage = dib.create_image(rbuf, m_bpp);
            }
            dib.put_image(dc, m_bimage, draw_x, draw_y, draw_width, draw_height);
            return;
        }
        
        // Optimization hint: make pmap_tmp as a private class member and reused it when possible.
        pixel_map pmap_tmp(rbuf->width(), rbuf->height(), m_sys_format, 256, m_flip_y);
        rendering_buffer* rbuf2 = &pmap_tmp.rbuf();
        
        switch(m_format)
        {
        case pix_format_rgb565:
            color_conv(rbuf2, rbuf, color_conv_rgb565_to_rgb555());
            break;
            
        case pix_format_rgb24:
            color_conv(rbuf2, rbuf, color_conv_rgb24_to_bgr24());
            break;
            
        case pix_format_abgr32:
            color_conv(rbuf2, rbuf, color_conv_abgr32_to_bgra32());
            break;
            
        case pix_format_argb32:
            color_conv(rbuf2, rbuf, color_conv_argb32_to_bgra32());
            break;
            
        case pix_format_rgba32:
            color_conv(rbuf2, rbuf, color_conv_rgba32_to_bgra32());
            break;
            
        case pix_format_gray8:
        case end_of_pix_formats: 
        case pix_format_undefined:
            ;
        }

        // This will ultimately call back to us, going to the top if branch since
        // the pix_format is compatible.
        pmap_tmp.draw(dc, draw_x, draw_y, draw_width, draw_height);
        
    }
    
    //------------------------------------------------------------------------
    unsigned platform_specific::calc_row_len(unsigned width, unsigned bits_per_pixel)
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

}

