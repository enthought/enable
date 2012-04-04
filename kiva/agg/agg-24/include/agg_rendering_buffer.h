//----------------------------------------------------------------------------
// Anti-Grain Geometry - Version 2.4
// Copyright (C) 2002-2005 Maxim Shemanarev (http://www.antigrain.com)
//
// Permission to copy, use, modify, sell and distribute this software 
// is granted provided this copyright notice appears in all copies. 
// This software is provided "as is" without express or implied
// warranty, and with no claim as to its suitability for any purpose.
//
//----------------------------------------------------------------------------
// Contact: mcseem@antigrain.com
//          mcseemagg@yahoo.com
//          http://www.antigrain.com
//----------------------------------------------------------------------------
//
// class rendering_buffer
//
//----------------------------------------------------------------------------

#ifndef AGG_RENDERING_BUFFER_INCLUDED
#define AGG_RENDERING_BUFFER_INCLUDED

#include "agg_array.h"

namespace agg24
{

    //==========================================================row_ptr_cache
    template<class T> class row_ptr_cache
    {
    public:
        //--------------------------------------------------------------------
        struct row_data
        {
            int x1, x2;
            const int8u* ptr;
            row_data() {}
            row_data(int x1_, int x2_, const int8u* ptr_) : 
                x1(x1_), x2(x2_), ptr(ptr_) {}
        };

        //-------------------------------------------------------------------
        row_ptr_cache() :
            m_buf(0),
            m_rows(),
            m_width(0),
            m_height(0),
            m_stride(0)
        {
        }

        //--------------------------------------------------------------------
        row_ptr_cache(T* buf, unsigned width, unsigned height, int stride) :
            m_buf(0),
            m_rows(),
            m_width(0),
            m_height(0),
            m_stride(0)
        {
            attach(buf, width, height, stride);
        }

        //--------------------------------------------------------------------
        void attach(T* buf, unsigned width, unsigned height, int stride)
        {
            m_buf = buf;
            m_width = width;
            m_height = height;
            m_stride = stride;
            if(height > m_rows.size())
            {
                m_rows.resize(height);
            }

            T* row_ptr = m_buf;

            if(stride < 0)
            {
                row_ptr = m_buf - int(height - 1) * (stride);
            }

            T** rows = &m_rows[0];

            while(height--)
            {
                *rows++ = row_ptr;
                row_ptr += stride;
            }
        }

        //--------------------------------------------------------------------
              T* buf()          { return m_buf;    }
        const T* buf()    const { return m_buf;    }
        unsigned width()  const { return m_width;  }
        unsigned height() const { return m_height; }
        int      stride() const { return m_stride; }
        unsigned stride_abs() const 
        {
            return (m_stride < 0) ? unsigned(-m_stride) : unsigned(m_stride); 
        }

        //--------------------------------------------------------------------
              T* row_ptr(int, int y, unsigned) { return m_rows[y]; }
              T* row_ptr(int y)                { return m_rows[y]; }
        const T* row_ptr(int y) const          { return m_rows[y]; }
        row_data row    (int y) const { return row_data(0, m_width-1, m_rows[y]); }

        //--------------------------------------------------------------------
        T const* const* rows() const { return &m_rows[0]; }

        //--------------------------------------------------------------------
        template<class RenBuf>
        void copy_from(const RenBuf& src)
        {
            unsigned h = height();
            if(src.height() < h) h = src.height();
        
            unsigned l = stride_abs();
            if(src.stride_abs() < l) l = src.stride_abs();
        
            l *= sizeof(T);

            unsigned y;
            unsigned w = width();
            for (y = 0; y < h; y++)
            {
                memcpy(row_ptr(0, y, w), src.row_ptr(y), l);
            }
        }

        //--------------------------------------------------------------------
        void clear(T value)
        {
            unsigned y;
            unsigned w = width();
            unsigned stride = stride_abs();
            for(y = 0; y < height(); y++)
            {
                T* p = row_ptr(0, y, w);
                unsigned x;
                for(x = 0; x < stride; x++)
                {
                    *p++ = value;
                }
            }
        }

    private:
        //--------------------------------------------------------------------
        T*            m_buf;        // Pointer to renrdering buffer
        pod_array<T*> m_rows;       // Pointers to each row of the buffer
        unsigned      m_width;      // Width in pixels
        unsigned      m_height;     // Height in pixels
        int           m_stride;     // Number of bytes per row. Can be < 0
    };



    //========================================================rendering_buffer
    typedef row_ptr_cache<int8u> rendering_buffer;

}


#endif
