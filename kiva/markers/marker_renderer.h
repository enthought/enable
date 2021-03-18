// (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
// All rights reserved.
//
// This software is provided without warranty under the terms of the BSD
// license included in LICENSE.txt and may be redistributed only under
// the conditions described in the aforementioned license. The license
// is also available online at http://www.enthought.com/licenses/BSD.txt
//
// Thanks for using Enthought open source!
#ifndef KIVA_MARKER_RENDERER_H
#define KIVA_MARKER_RENDERER_H

#include <agg_pixfmt_rgb.h>
#include <agg_pixfmt_rgba.h>
#include <agg_renderer_base.h>
#include <agg_renderer_markers.h>
#include <agg_trans_affine.h>

namespace kiva_markers
{
    // This enumeration must match the marker constants in `kiva.constants`!
    enum marker_type
    {
        MARKER_SQUARE = 1,
        MARKER_DIAMOND,
        MARKER_CIRCLE,
        MARKER_CROSSED_CIRCLE,
        MARKER_CROSS,
        MARKER_TRIANGLE,
        MARKER_INVERTED_TRIANGLE,
        MARKER_PLUS,
        MARKER_DOT,
        MARKER_PIXEL,
    };

    class marker_renderer_base
    {
    public:
        virtual ~marker_renderer_base(){}
        virtual bool draw_markers(const double* pts, const unsigned Npts,
                                  const unsigned size, const marker_type type,
                                  const double* fill, const double* stroke) = 0;
        virtual void transform(const double sx, const double sy, const double shx,
                               const double shy, const double tx, const double ty) = 0;
    };

    template<typename pixfmt_t>
    class marker_renderer : public marker_renderer_base
    {
    public:
        marker_renderer(unsigned char* buf,
                        const unsigned width, const unsigned height,
                        const int stride, const bool bottom_up = false)
        : m_renbuf(buf, width, height, bottom_up ? -stride : stride)
        , m_pixfmt(m_renbuf)
        , m_base_renderer(m_pixfmt)
        , m_renderer(m_base_renderer)
        {}

        virtual ~marker_renderer() {}

        bool draw_markers(const double* pts, const unsigned Npts,
                          const unsigned size, const marker_type type,
                          const double* fill, const double* stroke)
        {
            // Map from our marker type to the AGG marker type
            const agg24markers::marker_e marker = _get_marker_type(type);
            if (marker == agg24markers::end_of_markers) return false;

            // Assign fill and line colors
            m_renderer.fill_color(agg24markers::rgba(fill[0], fill[1], fill[2], fill[3]));
            m_renderer.line_color(agg24markers::rgba(stroke[0], stroke[1], stroke[2], stroke[3]));

            // NOTE: this is the average in X and Y
            const double scale = m_transform.scale();

            // Draw the markers
            double mx, my;
            for (unsigned i = 0; i < Npts*2; i+=2)
            {
                mx = pts[i];
                my = pts[i+1];
                m_transform.transform(&mx, &my);
                m_renderer.marker(int(mx), int(my), size * scale, marker);
            }

            return true;
        }

        void transform(const double sx, const double sy,
                       const double shx, const double shy,
                       const double tx, const double ty)
        {
            m_transform.sx = sx; m_transform.sy = sy;
            m_transform.shx = shx; m_transform.shy = shy;
            m_transform.tx = tx; m_transform.ty = ty;
        }

    private:
        agg24markers::marker_e _get_marker_type(const marker_type type) const
        {
            switch (type)
            {
                case MARKER_SQUARE: return agg24markers::marker_square;
                case MARKER_DIAMOND: return agg24markers::marker_diamond;
                case MARKER_CIRCLE: return agg24markers::marker_circle;
                case MARKER_CROSSED_CIRCLE: return agg24markers::marker_crossed_circle;
                case MARKER_CROSS: return agg24markers::marker_x;
                case MARKER_TRIANGLE: return agg24markers::marker_triangle_up;
                case MARKER_INVERTED_TRIANGLE: return agg24markers::marker_triangle_down;
                case MARKER_PLUS: return agg24markers::marker_cross;
                case MARKER_DOT: return agg24markers::marker_dot;
                case MARKER_PIXEL: return agg24markers::marker_pixel;
            }
            return agg24markers::end_of_markers;
        }

        typedef agg24markers::renderer_base<pixfmt_t> base_renderer_t;
        typedef agg24markers::renderer_markers<base_renderer_t> renderer_t;

        agg24markers::rendering_buffer m_renbuf;
        pixfmt_t m_pixfmt;
        base_renderer_t m_base_renderer;
        renderer_t m_renderer;
        agg24markers::trans_affine m_transform;
    };

} // namespace kiva_markers

#endif
