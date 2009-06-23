#ifndef KIVA_GRADIENT_H
#define KIVA_GRADIENT_H

#include <iostream>

#include <utility>
#include <vector>

#include "agg_pixfmt_rgb.h"
#include "agg_rasterizer_scanline_aa.h"
#include "agg_renderer_scanline.h"
#include "agg_scanline_u.h"
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
        gradient_spread_e spread_method;

        gradient(gradient_type_e gradient_type);
        gradient(gradient_type_e gradient_type,
                std::vector<point> points,
                std::vector<gradient_stop> stops,
                const char* spread_method);
        ~gradient();

        template <typename pixfmt_type>
        void apply(pixfmt_type pixfmt,
                   agg::rasterizer_scanline_aa<>* ras,
                   agg::renderer_mclip<pixfmt_type>* rbase)
        {
            if (this->gradient_type == kiva::grad_linear)
            {
                if (this->points[0].first == this->points[1].first)
                {
                    agg::gradient_y grad_func;
                    this->_apply(pixfmt, ras, rbase, grad_func);
                }
                else if (this->points[0].second == this->points[1].second)
                {
                    agg::gradient_x grad_func;
                    this->_apply(pixfmt, ras, rbase, grad_func);
                }
                else
                {
                    agg::gradient_xy grad_func;
                    this->_apply(pixfmt, ras, rbase, grad_func);
                }
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

            std::vector<point> user_space_points;

            double d1 = this->points[0].first;
            double d2 = this->points[1].first;

            if ((this->gradient_type == kiva::grad_radial) && (this->points.size() >2))
            {
                d1 = this->points[2].first;
                d2 = this->points[2].first + (this->points[1].first-this->points[0].first);

                this->_apply_linear_transform(points[0], points[1], gradient_mtx, d2);
            }
            else if (this->gradient_type == kiva::grad_linear)
            {
                // veritcal, horizontal, and point-to-point gradients are
                // special cased because each one needs a slightly different
                // set of transformations. Special casing should not be needed,
                // but better agg docs or a lot more time to read the agg source
                // would be required...
                if (points[0].first == points[1].first)
                {
                    // vertical special cased because atan2(dx, dy)
                    double dx = points[1].first - points[0].first;
                    double dy = points[1].second - points[0].second;
                    d1 = this->points[0].second;
                    d2 = this->points[1].second;
                    gradient_mtx *= agg::trans_affine_scaling(sqrt(dx * dx + dy * dy) / (d2-d1));
                    gradient_mtx *= agg::trans_affine_rotation(atan2(dx, dy));
                }
                else if (points[0].second == points[1].second)
                {
                    // no transforms necessary for horizontal
                }
                else
                {
                    // point-to-point special cased because the vector to apply the
                    // gradient to is the hypotenuse
                    double dx = points[1].first - points[0].first;
                    double dy = points[1].second - points[0].second;
                    gradient_mtx *= agg::trans_affine_scaling(sqrt(dx * dx + dy * dy) / (d2-d1));
                    gradient_mtx *= agg::trans_affine_translation(points[0].first, points[0].second);
                }
            }

            span_gradient_type span_gradient(span_interpolator,
                                            gradient_func,
                                            color_array,
                                            d1, d2);

            renderer_gradient_type grad_renderer(*rbase, span_allocator, span_gradient);


            this->fill_color_array(color_array);

            agg::render_scanlines(*ras, scanline, grad_renderer);
        }


        void _apply_linear_transform(point p1, point p2, agg::trans_affine& mtx, double d2);

        template <class Array>
        void fill_color_array(Array& array)
        {
            // The agg::rgb::gradient function is not documented, so here's
            // my guess at what it does: the first argument is obvious,
            // since we are constructing a gradient from one color to another.
            // The 2nd argument is a float, which must be between 0 and 1, and
            // represents the ratio of the first color to the second color.
            // Hence, it should always go from 0->1. In a multi-stop scenario
            // we will loop through the stops, for each pair the gradient call
            // will go from 0% to 100% of the 2nd color.
            // I hope that makes sense.

            std::vector<gradient_stop>::iterator stop_it = this->stops.begin();
            double offset = 0.0;
            unsigned int i = 0;

            for (; stop_it+1 != this->stops.end(); stop_it++)
            {
                std::vector<gradient_stop>::iterator next_it = stop_it+1;
                double offset_range = next_it->offset - stop_it->offset;
                while ( (offset <= next_it->offset) && (offset <=1.0))
                {
                    array[i] = stop_it->color.gradient(next_it->color, (offset-stop_it->offset)/offset_range);
                    i++;
                    offset = i/double(array.size());
                }
            }

            if (this->spread_method == kiva::pad)
            {
                for (; i < array.size(); i++)
                {
                    array[i] = this->stops.back().color;
                }
            }
            else if (this->spread_method == kiva::reflect)
            {
                // TODO: handle 'relect' spread method
            }
            else
            {
                // TODO: handle 'repeat' spread method
            }
        }
    };
}

#endif
