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
#ifndef AGG_WIN32_BMP_INCLUDED
#define AGG_WIN32_BMP_INCLUDED

#include "Python.h"
#include "win32/agg_platform_specific.h"

namespace agg24
{
    class pixel_map
    {
        public:
            pixel_map(unsigned width, unsigned height, pix_format_e format,
                      unsigned clear_val, bool bottom_up);
            ~pixel_map();
            void draw(HDC h_dc, int draw_x=-1, int draw_y=-1, int draw_width=-1,
            		  int draw_height=-1) const;
            pix_format_e get_pix_format() const;
        
            unsigned char* buf();
            unsigned       width() const;
            unsigned       height() const;
            int            stride() const;
            unsigned       bpp() const { return m_bpp; }
            rendering_buffer& rbuf() { return m_rbuf_window; }
            platform_specific*  m_specific;
            PyObject* convert_to_argb32string() const;
        
        private:
            void        destroy();
            void        create(unsigned width, 
                               unsigned height,
                               unsigned clear_val=256);
        
            unsigned char*   m_buf;
            unsigned         m_bpp;
            rendering_buffer m_rbuf_window;
        
        public:
        
    };

}


#endif
