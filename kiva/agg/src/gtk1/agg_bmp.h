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
#ifndef AGG_GTK1_BMP_INCLUDED
#define AGG_GTK1_BMP_INCLUDED
#include "Python.h"
#include "gtk1/agg_platform_specific.h"

namespace agg24
{

  class pixel_map
    {
    public:
      ~pixel_map();
      pixel_map(unsigned width, unsigned height, pix_format_e format,
		unsigned clear_val, bool bottom_up);
      void draw(Window h_dc, int x=0, int y=0, double scale=1.0) const;
      pix_format_e get_pix_format() const;

    public:
      unsigned char* buf();
      unsigned       width() const;
      unsigned       height() const;
      unsigned       stride() const;
      unsigned       bpp() const { return m_bpp; }
      rendering_buffer& rbuf() { return m_rbuf_window; }
      PyObject* convert_to_rgbarray() const;

    private:
        void        destroy();
        void        create(unsigned width,
                           unsigned height,
                           unsigned clear_val=256);
    private:
        Pixmap*        m_bmp;
        unsigned char* m_buf; // -> m_bmp
        unsigned       m_bpp; // calculated from format
        rendering_buffer m_rbuf_window;

    public:
	platform_specific*  m_specific;

    };

}

#endif
