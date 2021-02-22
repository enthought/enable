// (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
// All rights reserved.
//
// This software is provided without warranty under the terms of the BSD
// license included in LICENSE.txt and may be redistributed only under
// the conditions described in the aforementioned license. The license
// is also available online at http://www.enthought.com/licenses/BSD.txt
//
// Thanks for using Enthought open source!
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

#include "kiva_affine_helpers.h"
#include "kiva_constants.h"

namespace kiva
{
    class gradient_stop
    {
        public:
        double offset;
        agg24::rgba8 color;

        gradient_stop(double offset, agg24::rgba8& color) :
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
        gradient_units_e units;

        private:
        agg24::trans_affine affine_mtx;

        public:
        gradient(gradient_type_e gradient_type);
        gradient(gradient_type_e gradient_type,
                std::vector<point> points,
                std::vector<gradient_stop> stops,
                const char* spread_method,
                const char* units="userSpaceOnUse");
        ~gradient();

        template <typename pixfmt_type>
        void apply(pixfmt_type pixfmt,
                   agg24::rasterizer_scanline_aa<>* ras,
                   agg24::renderer_mclip<pixfmt_type>* rbase)
        {
            if (this->gradient_type == kiva::grad_linear)
            {
                if (this->points[0].first == this->points[1].first)
                {
                    agg24::gradient_y grad_func;

                    // apply the proper fill adapter based on the spread method

                    if (this->spread_method == kiva::reflect)
                    {
                        agg24::gradient_reflect_adaptor<agg24::gradient_y> adaptor(grad_func);
                        this->_apply(pixfmt, ras, rbase, adaptor);
                    }
                    else if (this->spread_method == kiva::repeat)
                    {
                        agg24::gradient_repeat_adaptor<agg24::gradient_y> adaptor(grad_func);
                        this->_apply(pixfmt, ras, rbase, adaptor);
                    }
                    else
                    {
                        this->_apply(pixfmt, ras, rbase, grad_func);
                    }
                }
                else if (this->points[0].second == this->points[1].second)
                {
                    agg24::gradient_x grad_func;

                    // apply the proper fill adapter based on the spread method

                    if (this->spread_method == kiva::reflect)
                    {
                        agg24::gradient_reflect_adaptor<agg24::gradient_x> adaptor(grad_func);
                        this->_apply(pixfmt, ras, rbase, adaptor);
                    }
                    else if (this->spread_method == kiva::repeat)
                    {
                        agg24::gradient_repeat_adaptor<agg24::gradient_x> adaptor(grad_func);
                        this->_apply(pixfmt, ras, rbase, adaptor);
                    }
                    else
                    {
                        this->_apply(pixfmt, ras, rbase, grad_func);
                    }
                }
                else
                {
                    agg24::gradient_x grad_func;

                    // apply the proper fill adapter based on the spread method

                    if (this->spread_method == kiva::reflect)
                    {
                        agg24::gradient_reflect_adaptor<agg24::gradient_x> adaptor(grad_func);
                        this->_apply(pixfmt, ras, rbase, adaptor);
                    }
                    else if (this->spread_method == kiva::repeat)
                    {
                        agg24::gradient_repeat_adaptor<agg24::gradient_x> adaptor(grad_func);
                        this->_apply(pixfmt, ras, rbase, adaptor);
                    }
                    else
                    {
                        this->_apply(pixfmt, ras, rbase, grad_func);
                    }
                }
            }
            else
            {
                agg24::gradient_radial_focus grad_func(points[1].first,
                                                     points[2].first - points[0].first,
                                                     points[2].second - points[0].second);

                if (this->spread_method == kiva::reflect)
                {
                    agg24::gradient_reflect_adaptor<agg24::gradient_radial_focus> adaptor(grad_func);
                    this->_apply(pixfmt, ras, rbase, adaptor);
                }
                else if (this->spread_method == kiva::repeat)
                {
                    agg24::gradient_repeat_adaptor<agg24::gradient_radial_focus> adaptor(grad_func);
                    this->_apply(pixfmt, ras, rbase, adaptor);
                }
                else
                {
                    this->_apply(pixfmt, ras, rbase, grad_func);
                }
            }
        }

        void set_ctm(const agg24::trans_affine& mtx)
        {
            this->affine_mtx = mtx;
        }

        protected:

        template <class pixfmt_type, class gradient_func_type>
        void _apply(pixfmt_type pixfmt,
                   agg24::rasterizer_scanline_aa<>* ras,
                   agg24::renderer_mclip<pixfmt_type>* rbase,
                   gradient_func_type gradient_func)
        {
            typedef agg24::renderer_mclip<pixfmt_type> renderer_base_type;
            typedef agg24::span_interpolator_linear<> interpolator_type;
            typedef agg24::span_allocator<agg24::rgba8> span_allocator_type;
            typedef agg24::pod_auto_array<agg24::rgba8, 256> color_array_type;
            typedef agg24::span_gradient<agg24::rgba8,
                                    interpolator_type,
                                    gradient_func_type,
                                    color_array_type> span_gradient_type;
            typedef agg24::renderer_scanline_aa<renderer_base_type,
                                                span_allocator_type,
                                                span_gradient_type> renderer_gradient_type;


            agg24::trans_affine   gradient_mtx;                    // Affine transformer
            interpolator_type   span_interpolator(gradient_mtx); // Span interpolator
            span_allocator_type span_allocator;                  // Span Allocator
            color_array_type    color_array;                     // Gradient colors
            agg24::scanline_u8 scanline;

            double dx = points[1].first - points[0].first;
            double dy = points[1].second - points[0].second;
            double d1 = 0, d2 = 0;

            if ((this->gradient_type == kiva::grad_radial) && (this->points.size() >2))
            {
                // length is the radius
                d2 = points[1].first;
            }
            else if (this->gradient_type == kiva::grad_linear)
            {
                // length is the distance between the two points
                d2 = sqrt(dx * dx + dy * dy);
                
                if (points[0].first == points[1].first)
                {
                    // gradient_y. handle flips
                    gradient_mtx *= agg24::trans_affine_rotation(atan2(0.0, dy));
                }
                else if (points[0].second == points[1].second)
                {
                    // gradient_x. handle flips
                    gradient_mtx *= agg24::trans_affine_rotation(atan2(0.0, dx));
                }
                else
                {
                    // general case: arbitrary rotation
                    gradient_mtx *= agg24::trans_affine_rotation(atan2(dy, dx));
                }
            }

            gradient_mtx *= agg24::trans_affine_translation(points[0].first, points[0].second);
            if (this->units == kiva::user_space)
            {
                gradient_mtx *= this->affine_mtx;
            }
            gradient_mtx.invert();

            //std::cout << "drawing with affine matrix " << gradient_mtx.m0
            //                                           << ", " << gradient_mtx.m1
            //                                           << ", " << gradient_mtx.m2
            //                                           << ", " << gradient_mtx.m3
            //                                           << ", " << gradient_mtx.m4
            //                                           << ", " << gradient_mtx.m5 << std::endl;

            span_gradient_type span_gradient(span_interpolator,
                                             gradient_func,
                                             color_array,
                                             d1, d2);

            renderer_gradient_type grad_renderer(*rbase, span_allocator, span_gradient);


            this->fill_color_array(color_array);

            agg24::render_scanlines(*ras, scanline, grad_renderer);
        }


        template <class Array>
        void fill_color_array(Array& array)
        {
            // The agg24::rgb::gradient function is not documented, so here's
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
            unsigned int const array_size = array.size();

            for (; stop_it+1 != this->stops.end(); stop_it++)
            {
                std::vector<gradient_stop>::iterator next_it = stop_it+1;
                double offset_range = next_it->offset - stop_it->offset;
                while ( (offset <= next_it->offset) && (i < array_size))
                {
                    array[i++] = stop_it->color.gradient(next_it->color, (offset-stop_it->offset)/offset_range);
                    offset = i/double(array_size-1);
                }
            }
        }
    };
}

#endif
