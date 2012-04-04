#ifndef KIVA_IMAGE_FILTERS_H
#define KIVA_IMAGE_FILTERS_H

#include "agg_pixfmt_rgb.h"
#include "agg_pixfmt_rgba.h"

#include "agg_span_image_filter_rgba.h"
#include "agg_span_image_filter_rgb.h"

#include "agg_image_accessors.h"

#include "agg_span_interpolator_linear.h"

namespace kiva
{

    typedef agg24::span_interpolator_linear<> interpolator_type;
    
    template<class pixel_format>
    class image_filters
    {
    };
    
    template<>
    class image_filters<agg24::pixfmt_rgba32>
    {
        public:
        typedef agg24::image_accessor_clip<agg24::pixfmt_rgba32> source_type;
        
        typedef agg24::span_image_filter_rgba_nn<source_type, 
                                           interpolator_type> nearest_type;
        typedef agg24::span_image_filter_rgba_bilinear<source_type, 
                                           interpolator_type> bilinear_type;
        typedef agg24::span_image_filter_rgba<source_type, 
                                           interpolator_type> general_type;                   
    };

    template<>
    class image_filters<agg24::pixfmt_bgra32>
    {
        public:
        typedef agg24::image_accessor_clip<agg24::pixfmt_bgra32> source_type;
        
        typedef agg24::span_image_filter_rgba_nn<source_type, 
                                           interpolator_type> nearest_type;
        typedef agg24::span_image_filter_rgba_bilinear<source_type, 
                                           interpolator_type> bilinear_type;
        typedef agg24::span_image_filter_rgba<source_type, 
                                           interpolator_type> general_type;                   
    };
    
    template<>
    class image_filters<agg24::pixfmt_argb32>
    {
        public:
        typedef agg24::image_accessor_clip<agg24::pixfmt_argb32> source_type;
        
        typedef agg24::span_image_filter_rgba_nn<source_type, 
                                           interpolator_type> nearest_type;
        typedef agg24::span_image_filter_rgba_bilinear<source_type, 
                                           interpolator_type> bilinear_type;
        typedef agg24::span_image_filter_rgba<source_type, 
                                           interpolator_type> general_type;                   
    };
    
    template<>
    class image_filters<agg24::pixfmt_abgr32>
    {
        public:
        typedef agg24::image_accessor_clip<agg24::pixfmt_abgr32> source_type;
        
        typedef agg24::span_image_filter_rgba_nn<source_type, 
                                           interpolator_type> nearest_type;
        typedef agg24::span_image_filter_rgba_bilinear<source_type, 
                                           interpolator_type> bilinear_type;
        typedef agg24::span_image_filter_rgba<source_type, 
                                           interpolator_type> general_type;                   
    };
    
    template<>
    class image_filters<agg24::pixfmt_rgb24>
    {
        public:
        typedef agg24::image_accessor_clip<agg24::pixfmt_rgb24> source_type;
        
        typedef agg24::span_image_filter_rgb_nn<source_type, 
                                           interpolator_type> nearest_type;
        typedef agg24::span_image_filter_rgb_bilinear<source_type, 
                                           interpolator_type> bilinear_type;
        typedef agg24::span_image_filter_rgb<source_type, 
                                           interpolator_type> general_type;                   
    };
    
    template<>
    class image_filters<agg24::pixfmt_bgr24>
    {
        public:
        typedef agg24::image_accessor_clip<agg24::pixfmt_bgr24> source_type;
        
        typedef agg24::span_image_filter_rgb_nn<source_type, 
                                           interpolator_type> nearest_type;
        typedef agg24::span_image_filter_rgb_bilinear<source_type, 
                                           interpolator_type> bilinear_type;
        typedef agg24::span_image_filter_rgb<source_type, 
                                           interpolator_type> general_type;                   
    };
}

#endif
