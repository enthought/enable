#ifndef KIVA_GRADIENT_H
#define KIVA_GRADIENT_H

#include <iostream>

#include <utility>
#include <vector>

#include "agg_pixfmt_rgb.h"
#include "agg_renderer_scanline.h"
#include "agg_span_gradient.h"
#include "agg_span_allocator.h"
#include "agg_span_interpolator_linear.h"
#include "agg_renderer_mclip.h"

#include "kiva_constants.h"

namespace kiva
{
    class gradient
    {
        public:
        typedef std::pair<double, double> point;
        typedef std::pair<double, agg::rgba8> stop;

        std::vector<point> points;
        std::vector<stop> stops;
        gradient_type_e gradient_type;

        gradient(gradient_type_e gradient_type) :
            gradient_type(gradient_type)
        {
        }

        gradient(gradient_type_e gradient_type, std::vector<point> points, std::vector<stop> stops) :
            points(points),
            stops(stops),
            gradient_type(gradient_type)
        {
        }

        ~gradient()
        {
        }

        template <typename pixfmt_type>
        void apply(pixfmt_type pixfmt,
                   agg::rasterizer_scanline_aa<>* ras,
                   agg::renderer_mclip<pixfmt_type>* rbase)
        {
            if (this->gradient_type == kiva::grad_linear)
            {
                agg::gradient_x grad_func;
                this->_apply(pixfmt, ras, rbase, grad_func);
            }
            else
            {
                agg::gradient_circle grad_func;
                this->_apply(pixfmt, ras, rbase, grad_func);
            }
        }

        protected:

        template <class pixfmt_type, class gradient_func_type>
        void _apply(pixfmt_type pixfmt,
                   agg::rasterizer_scanline_aa<>* ras,
                   agg::renderer_mclip<pixfmt_type>* rbase,
                   gradient_func_type gradient_func)
        {
            typedef agg::renderer_mclip<pixfmt_type> renderer_base_type;
            typedef agg::span_interpolator_linear<> interpolator_type;
            typedef agg::span_allocator<agg::rgba8> span_allocator_type;
            typedef agg::pod_auto_array<agg::rgba8, 256> color_array_type;
            typedef agg::span_gradient<agg::rgba8,
                                    interpolator_type,
                                    gradient_func_type,
                                    color_array_type> span_gradient_type;
            typedef agg::renderer_scanline_aa<renderer_base_type,
                                                span_allocator_type,
                                                span_gradient_type> renderer_gradient_type;


            agg::trans_affine   gradient_mtx;                    // Affine transformer
            interpolator_type   span_interpolator(gradient_mtx); // Span interpolator
            span_allocator_type span_allocator;                  // Span Allocator
            color_array_type    color_array;                     // Gradient colors
            agg::scanline_u8 scanline;

            double d1 = this->points[0].first;
            double d2 = this->points[1].first;

            if ((this->gradient_type == kiva::grad_radial) && (this->points.size() >2))
            {
                d1 = points[2].first;
                d2 = points[2].first + (points[1].first-points[0].first);
            }

            this->_apply_linear_transform(points[0], points[1], gradient_mtx, d2);

            std::cout << "applying gradient" << std::endl;
            span_gradient_type span_gradient(span_interpolator,
                                            gradient_func,
                                            color_array,
                                            d1, d2);

            // TODO: use the offsets
            this->fill_two_stop_array(color_array, this->stops[0].second, this->stops[1].second);

            renderer_gradient_type grad_renderer(*rbase, span_allocator, span_gradient);
            agg::render_scanlines(*ras, scanline, grad_renderer);
        }

        void _apply_linear_transform(point p1, point p2, agg::trans_affine& mtx, double d2)
        {
            double dx = p2.first - p1.first;
            double dy = p2.second - p1.second;
            mtx.reset();
            mtx *= agg::trans_affine_scaling(sqrt(dx * dx + dy * dy) / d2);
            mtx *= agg::trans_affine_rotation(atan2(dy, dx));
            mtx *= agg::trans_affine_translation(p1.first, p1.second);
            mtx.invert();
        }

        // A simple function to form the gradient color array
        // consisting of 2 colors, "begin", "end"
        //---------------------------------------------------
        template<class Array>
        void fill_two_stop_array(Array& array,
                            agg::rgba8 begin,
                            agg::rgba8 end)
        {
            unsigned i;
            for(i = 0; i < array.size(); ++i)
            {
                array[i] = begin.gradient(end, i / double(array.size()));
            }
        }

        // A simple function to form the gradient color array
        // consisting of 3 colors, "begin", "middle", "end"
        //---------------------------------------------------
        template<class Array>
        void fill_three_stop_array(Array& array,
                            agg::rgba8 begin,
                            agg::rgba8 middle,
                            agg::rgba8 end)
        {
            unsigned i;
            unsigned half_size = array.size() / 2;
            for(i = 0; i < half_size; ++i)
            {
                array[i] = begin.gradient(middle, i / double(half_size));
            }
            for(; i < array.size(); ++i)
            {
                array[i] = middle.gradient(end, (i - half_size) / double(half_size));
            }
        }


    };
}

#endif