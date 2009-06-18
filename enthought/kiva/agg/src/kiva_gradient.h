#ifndef KIVA_GRADIENT_H
#define KIVA_GRADIENT_H

#include <iostream>

#include <utility>
#include <vector>

#include "agg_pixfmt_rgb.h"
#include "agg_rasterizer_scanline_aa.h"
#include "agg_span_gradient.h"
#include "agg_span_allocator.h"
#include "agg_span_interpolator_linear.h"
#include "agg_renderer_mclip.h"

#include "kiva_constants.h"

namespace kiva
{
    class gradient_stop
    {
        public:
        double offset;
        agg::rgba8 color;

        gradient_stop(double offset, agg::rgba8& color) :
            offset(offset),
            color(color)
        {
        }
    };

    class gradient
    {
        public:
        typedef std::pair<double, double> point;

        std::vector<point> points;
        std::vector<gradient_stop> stops;
        gradient_type_e gradient_type;

        gradient(gradient_type_e gradient_type);
        gradient(gradient_type_e gradient_type, std::vector<point> points, std::vector<gradient_stop> stops);
        ~gradient();

        template <typename pixfmt_type>
        void apply(pixfmt_type pixfmt,
                   agg::rasterizer_scanline_aa<>* ras,
                   agg::renderer_mclip<pixfmt_type>* rbase);

        protected:

        template <class pixfmt_type, class gradient_func_type>
        void _apply(pixfmt_type pixfmt,
                   agg::rasterizer_scanline_aa<>* ras,
                   agg::renderer_mclip<pixfmt_type>* rbase,
                   gradient_func_type gradient_func);

        void _apply_linear_transform(point p1, point p2, agg::trans_affine& mtx, double d2);

        template <class Array>
        void fill_color_array(Array& array);
  };
}

#endif