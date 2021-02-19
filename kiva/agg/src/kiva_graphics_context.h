// (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
// All rights reserved.
//
// This software is provided without warranty under the terms of the BSD
// license included in LICENSE.txt and may be redistributed only under
// the conditions described in the aforementioned license. The license
// is also available online at http://www.enthought.com/licenses/BSD.txt
//
// Thanks for using Enthought open source!
#ifndef KIVA_GRAPHICS_CONTEXT_H
#define KIVA_GRAPHICS_CONTEXT_H

#ifdef _MSC_VER
// Turn off MSDEV warning about truncated long identifiers
#pragma warning(disable:4786)
#endif

#include <assert.h>
#include <string.h>
#include <stack>

#include <iostream>
#include <memory>

#include "utf8.h"

#include "agg_trans_affine.h"
#include "agg_path_storage.h"

#include "agg_conv_stroke.h"
#include "agg_conv_dash.h"
#include "agg_conv_curve.h"
#include "agg_conv_clip_polygon.h"
#include "agg_conv_clip_polyline.h"

#include "agg_span_allocator.h"
#include "agg_span_converter.h"
#include "kiva_image_filters.h"

#include "agg_scanline_u.h"
#include "agg_scanline_bin.h"
#include "agg_scanline_p.h"

#include "agg_renderer_mclip.h"
#include "agg_renderer_scanline.h"
#include "agg_renderer_outline_aa.h"
#include "agg_renderer_primitives.h"

#include "agg_rasterizer_outline.h"
#include "agg_rasterizer_outline_aa.h"
#include "agg_rasterizer_scanline_aa.h"

#include "agg_span_gradient.h"

#include "kiva_image_filters.h"

#include "kiva_dash_type.h"
#include "kiva_compiled_path.h"
#include "kiva_font_type.h"
#include "kiva_pix_format.h"
#include "kiva_exceptions.h"
#include "kiva_graphics_context_base.h"
#include "kiva_alpha_gamma.h"
#include "kiva_gradient.h"


namespace kiva
{

    template <class agg_pixfmt>
    class graphics_context : public graphics_context_base
    {
        // agg_pixfmt has low level rendering commands (hspan, etc.)
        // This really should be a template parameter
        //typedef agg24::pixfmt_bgra32 agg_pixfmt;
        public:
        agg_pixfmt renderer_pixfmt;

        private:
        // The next level of renderer adds clipping to the low level
        // pixfmt object (renderer).  This is what attaches to a
        // rendering buffer.
        // The multi-clip renderer uses the single clip inside it,
        // so there is no need to define both and do our own switching.
        //typedef agg24::renderer_base<agg_pixfmt> renderer_base_type;
        typedef agg24::renderer_mclip<agg_pixfmt> renderer_base_type;

        renderer_base_type renderer;

        public:

        //---------------------------------------------------------------
        // constructor
        //---------------------------------------------------------------
        graphics_context(unsigned char *data,
                         int width, int height, int stride,
                         kiva::interpolation_e interp=nearest);


        ~graphics_context()
        {
        }
        // TODO: write copy constructor

        kiva::pix_format_e format();

        void restore_state();

        //---------------------------------------------------------------
        // Clipping path manipulation
        //---------------------------------------------------------------
        // Clips to the current path.  If the current path is empty,
        // does nothing.
        void clip();
        void even_odd_clip();

        // clip_to_rect() expects an array of doubles (x1,y1,x2,y2);
        // clip_to_rects() expects a repeated array of these.
        void clip_to_rect(double x, double y, double sx, double sy);
        void clip_to_rect(kiva::rect_type &rect);

        // Computes the (possibly disjoint) union of the input rectangles
        // and clips to them.  NOTE that this is not the same thing as
        // simply calling clip_to_rect() on each rectangle separately!
        void clip_to_rects(double* new_rects, int Nrects);
        void clip_to_rects(kiva::rect_list_type &rects);

        void clear_clip_path();

        int get_num_clip_regions();
        kiva::rect_type get_clip_region(unsigned int i);


        //---------------------------------------------------------------
        // Painting paths (drawing and filling contours)
        //---------------------------------------------------------------
        void clear(agg24::rgba value=agg24::rgba(1,1,1,1));
        void clear(double alpha);

        void draw_path(draw_mode_e mode=FILL_STROKE);
        void draw_rect(double rect[4],
                       draw_mode_e mode=FILL_STROKE);

        int draw_marker_at_points(double* pts,int Npts,int size,
                                   agg24::marker_e type=agg24::marker_square);

        void draw_path_at_points(double* pts,int Npts,
                                  kiva::compiled_path& marker,
                                  draw_mode_e mode);

        //---------------------------------------------------------------
        // Text handling
        //---------------------------------------------------------------

        bool show_text(char *text);


        //---------------------------------------------------------------
        // Image handling
        //---------------------------------------------------------------


        void draw_glyphs(kiva::graphics_context_base* img, double tx, double ty);

        int draw_image(kiva::graphics_context_base* img, double rect[4], bool force_copy=false);

        private:
        int blend_image(kiva::graphics_context_base* img, int tx, int ty);
        int copy_image(kiva::graphics_context_base* img, int tx, int ty);
        int transform_image(kiva::graphics_context_base* img,
                            agg24::trans_affine& img_mtx);

        private:
        // Transforms a clipping rectangle into device coordinates.
        kiva::rect_type transform_clip_rectangle(const kiva::rect_type &rect);

        public:

        //--------------------------------------------------------------------
        // Stroke Path Pipeline.
        //
        // See implementation_notes.txt for details.
        //--------------------------------------------------------------------

        void stroke_path()
        {
            this->_stroke_path();
            this->path.remove_all();
        }

        private:
        void _stroke_path()
        {
            // 1. Choose whether to do a curve conversion or not.
            // 2. Pick whether the line is dashed.

            // short circuit for transparent or 0 width lines
            if (this->state.line_color.a == 0 || this->state.line_width == 0.0)
        		return;

            if (!this->path.has_curves())
            {
                this->stroke_path_dash_conversion(this->path);
            }
            else
            {
                agg24::conv_curve<kiva::compiled_path> curved_path(this->path);
                this->stroke_path_dash_conversion(curved_path);
            }
        }

        private:
        template <class path_type>
        void stroke_path_dash_conversion(path_type& input_path)
        {
            if (this->state.line_dash.is_solid())
            {
                this->stroke_path_choose_clipping_renderer(input_path);
            }
            else
            {
                agg24::conv_dash<path_type> dashed_path(input_path);
                std::vector<double> &pattern = this->state.line_dash.pattern;
                // pattern always has even length
                for(unsigned int i = 0; i < pattern.size(); i+=2)
                {
                    dashed_path.add_dash(pattern[i], pattern[i+1]);
                }
                dashed_path.dash_start(this->state.line_dash.phase);

                this->stroke_path_choose_clipping_renderer(dashed_path);
            }
        }

       	private:
       	template<class path_type>
    	void stroke_path_choose_clipping_renderer(path_type& input_path)
    	{
            agg24::conv_clip_polyline<path_type> clipped(input_path);

            // fix me: We can do this more intelligently based on the current clip path.
            // fix me: What coordinates should this be in?  I think in user space instead
            //         of device space.  This looks wrong...
            clipped.clip_box(0,0, this->buf.width(), this->buf.height());

    	    // fix me: We should be doing a vector clip of the path as well.
    	    //         where is this step?

    	    // fix me: pick the renderer type (clipping, etc.)
    	    if(1)
    	    {
    	        this->stroke_path_choose_rasterizer(clipped, this->renderer);
    	    }
    	    else
    	    {
    	        // fix me: pick the renderer type (clipping, etc.)
    	    }
        }

       	private:
       	template<class path_type, class renderer_type>
    	void stroke_path_choose_rasterizer(path_type& input_path,
    	                                   renderer_type& input_renderer)
    	{
    		if (!this->state.should_antialias)
    		{
    			if ( this->state.line_width <= 1.0)
    			{
    				// ignore cap and join type here.
    				this->stroke_path_outline(input_path, input_renderer);
    			}

                // 2005-04-01: the AGG outline_aa rasterizer has a bug in it;
                // Go with the slower scanline_aa rasterizer until it's fixed
                // in AGG.

//    			else if ( this->state.line_width <=10.0 &&
//    					  (this->state.line_cap == CAP_ROUND ||
//    					   this->state.line_cap == CAP_BUTT)   &&
//    					  this->state.line_join == JOIN_MITER)
//    			{
//    				// fix me: how to force this to be aliased???
//    				this->stroke_path_outline_aa(input_path, input_renderer);
//    			}
    			else
    			{
    				// fix me: This appears to be anti-aliased still.
    				typedef agg24::renderer_scanline_bin_solid<renderer_type> renderer_bin_type;
    				renderer_bin_type renderer(input_renderer);
    			    agg24::scanline_bin scanline;

    				this->stroke_path_scanline_aa(input_path, renderer, scanline);
    			}
    		}
    		else // anti-aliased
    		{
//    			if ( (this->state.line_cap == CAP_ROUND || this->state.line_cap == CAP_BUTT) &&
//    				  this->state.line_join == JOIN_MITER)
//    			{
//    				this->stroke_path_outline_aa(input_path, input_renderer);
//    			}
//    			else
//    			{
    				typedef agg24::renderer_scanline_aa_solid<renderer_type> renderer_aa_type;
    				renderer_aa_type renderer(input_renderer);
    				agg24::scanline_u8 scanline;

    				this->stroke_path_scanline_aa(input_path, renderer, scanline);
//    			}
    		}

    	}

        private:
    	template<class path_type, class renderer_type>
    	void stroke_path_outline(path_type& input_path, renderer_type& input_renderer)
    	{
    		typedef agg24::renderer_primitives<renderer_type> primitives_renderer_type;
    		typedef agg24::rasterizer_outline<primitives_renderer_type> rasterizer_type;

    		primitives_renderer_type primitives_renderer(input_renderer);

    		// set line color -- multiply by alpha if it is set.
            agg24::rgba color;
            color = this->state.line_color;
            color.a *= this->state.alpha;

            primitives_renderer.line_color(color);
    		rasterizer_type rasterizer(primitives_renderer);
    		rasterizer.add_path(input_path);
    	}

    	private:
    	template<class path_type, class renderer_type>
    	void stroke_path_outline_aa(path_type& input_path, renderer_type& input_renderer)
    	{
    		// fix me: How do you render aliased lines with this?

    		// rasterizer_outline_aa algorithm only works for
    		// CAP_ROUND or CAP_BUTT.  It also only works for JOIN_MITER

    		typedef agg24::renderer_outline_aa<renderer_type> outline_renderer_type;
    		typedef agg24::rasterizer_outline_aa<outline_renderer_type> rasterizer_type;

    		// fix me: scale width by ctm
    		agg24::line_profile_aa profile(this->state.line_width, agg24::gamma_none());

    	    outline_renderer_type renderer(input_renderer, profile);

            // set line color -- multiply by alpha if it is set.
            agg24::rgba color;
            color = this->state.line_color;
            color.a *= this->state.alpha;
    		renderer.color(color);

    		rasterizer_type rasterizer(renderer);

            if (this->state.line_cap == CAP_ROUND)
            {
    	        rasterizer.round_cap(true);
            }
            else if (this->state.line_cap == CAP_BUTT)
            {    //default behavior
            }

    		// fix me: not sure about the setting for this...
    		rasterizer.accurate_join(true);

    		rasterizer.add_path(input_path);
    	}

    	private:
    	template<class path_type, class renderer_type, class scanline_type>
    	void stroke_path_scanline_aa(path_type& input_path, renderer_type& renderer,
    								 scanline_type& scanline)
    	{
    		agg24::rasterizer_scanline_aa<> rasterizer;

    		agg24::conv_stroke<path_type> stroked_path(input_path);

    		// fix me: scale width by ctm
    		stroked_path.width(this->state.line_width);

            // handle line cap
            agg24::line_cap_e cap = agg24::butt_cap;
            if (this->state.line_cap == CAP_ROUND)
            {
                cap = agg24::round_cap;
            }
            else if (this->state.line_cap == CAP_BUTT)
            {
                cap = agg24::butt_cap;
            }
            else if (this->state.line_cap == CAP_SQUARE)
            {
                cap = agg24::square_cap;
            }
            stroked_path.line_cap(cap);

            // handle join
            agg24::line_join_e join = agg24::miter_join;
            if (this->state.line_join == JOIN_MITER)
            {
                join = agg24::miter_join;
            }
            else if (this->state.line_join == JOIN_ROUND)
            {
                join = agg24::round_join;
            }
            else if (this->state.line_join == JOIN_BEVEL)
            {
                join = agg24::bevel_join;
            }
            stroked_path.line_join(join);

            // set line color -- multiply by alpha if it is set.
            agg24::rgba color;
            color = this->state.line_color;
            color.a *= this->state.alpha;
    		renderer.color(color);

    		// render
    		rasterizer.add_path(stroked_path);
    		agg24::render_scanlines(rasterizer, scanline, renderer);
    	}

        //--------------------------------------------------------------------
        // Fill Path Pipeline.
        //
        // See implementation_notes.txt for details.
        //--------------------------------------------------------------------

        public:
        void fill_path()
        {
            this->_fill_path(agg24::fill_non_zero);
            this->path.remove_all();
        }

        public:
        void eof_fill_path()
        {
            this->_fill_path(agg24::fill_even_odd);
            this->path.remove_all();
        }

        private:
        void _fill_path(agg24::filling_rule_e rule)
        {
            // 1. Choose whether to do a curve conversion or not.
            // 2. Pick whether the line is dashed.

            // short circuit for transparent
            if (this->state.fill_color.a == 0)
        		return;

            if (!this->path.has_curves())
            {
                this->fill_path_clip_conversion(this->path, rule);
            }
            else
            {
                agg24::conv_curve<kiva::compiled_path> curved_path(this->path);
                this->fill_path_clip_conversion(curved_path, rule);
            }
        }

        //---------------------------------------------------------------------
        // Gradient support
        //---------------------------------------------------------------------
        void linear_gradient(double x1, double y1,
                            double x2, double y2,
                            std::vector<kiva::gradient_stop> stops,
                            const char* spread_method,
                            const char* units="userSpaceOnUse")
        {
            typedef std::pair<double, double> point_type;
            std::vector<gradient_stop> stops_list;
            std::vector<point_type> points;
            
            if (strcmp(units, "objectBoundingBox") == 0)
            {
                // Transform from relative coordinates
                kiva::rect_type const clip_rect = _get_path_bounds();
                x1 = clip_rect.x + x1 * clip_rect.w;
                x2 = clip_rect.x + x2 * clip_rect.w;
                y1 = clip_rect.y + y1 * clip_rect.h;
                y2 = clip_rect.y + y2 * clip_rect.h;                
            }

            points.push_back(point_type(x1, y1));
            points.push_back(point_type(x2, y2));

            this->state.gradient_fill = gradient(kiva::grad_linear, points,
												stops, spread_method, units);
            this->state.gradient_fill.set_ctm(this->get_ctm());
        }

        void radial_gradient(double cx, double cy, double r,
                            double fx, double fy,
                            std::vector<kiva::gradient_stop> stops,
                            const char* spread_method,
                            const char* units="userSpaceOnUse")
        {
            typedef std::pair<double, double> point_type;
            std::vector<point_type> points;

            if (strcmp(units, "objectBoundingBox") == 0)
            {
                // Transform from relative coordinates
                kiva::rect_type const clip_rect = _get_path_bounds();
                r = r * clip_rect.w;
                cx = clip_rect.x + cx * clip_rect.w;
                fx = clip_rect.x + fx * clip_rect.w;
                cy = clip_rect.y + cy * clip_rect.h;
                fy = clip_rect.y + fy * clip_rect.h;
            }

            points.push_back(point_type(cx, cy));
            points.push_back(point_type(r, 0));
            points.push_back(point_type(fx, fy));

            this->state.gradient_fill = gradient(kiva::grad_radial, points,
												stops, spread_method, units);
            this->state.gradient_fill.set_ctm(this->get_ctm());
        }


        private:

        template <class path_type>
        void fill_path_clip_conversion(path_type& input_path,
                                       agg24::filling_rule_e rule)
        {
            // !! non-clipped version is about 8% faster or so for lion if it
            // !! is entirely on the screen.  It is slower, however, when
            // !! things are rendered off screen.  Perhaps we should add a
            // !! compiled_path method for asking path what its bounding box
            // !! is and call this if it is all within the screen.
            //rasterizer.add_path(this->path);
            agg24::conv_clip_polygon<path_type> clipped(input_path);
            clipped.clip_box(0,0, this->buf.width(), this->buf.height());

            // fix me: We can do this more intelligently based on the current clip path.
            // fix me: What coordinates should this be in?  I think in user space instead
            //         of device space.  This looks wrong...

            agg24::rasterizer_scanline_aa<> rasterizer;

            rasterizer.filling_rule(rule);
            rasterizer.add_path(clipped);

            if (this->state.gradient_fill.gradient_type == kiva::grad_none)
            {
                agg24::scanline_u8 scanline;

                // fix me: we need to select the renderer in another method.
                agg24::renderer_scanline_aa_solid< renderer_base_type >
                            aa_renderer(this->renderer);

                // set fill color -- multiply by alpha if it is set.
                agg24::rgba color;
                color = this->state.fill_color;
                color.a *= this->state.alpha;

                aa_renderer.color(color);
                // draw the filled path to the buffer
                agg24::render_scanlines(rasterizer, scanline, aa_renderer);
            }
            else
            {
                this->state.gradient_fill.apply(this->renderer_pixfmt,
                                                &rasterizer, &this->renderer);
            }
        }


        //---------------------------------------------------------------
        // Handle drawing filled rect quickly in some cases.
        //---------------------------------------------------------------

        private:
        int _draw_rect_simple(double rect[4],
                              draw_mode_e mode=FILL_STROKE);

        //---------------------------------------------------------------
        // Draw_image pipeline
        //---------------------------------------------------------------

        private:
        template<class other_format>
        void transform_image_interpolate(kiva::graphics_context<other_format>& img,
                        agg24::trans_affine& img_mtx)
        {


            agg24::path_storage img_outline = img.boundary_path(img_mtx);
            other_format src_pix(img.rendering_buffer());

            agg24::trans_affine inv_img_mtx = img_mtx;
            inv_img_mtx.invert();
            agg24::span_interpolator_linear<> interpolator(inv_img_mtx);

            agg24::rgba back_color = agg24::rgba(1,1,1,0);
            agg24::span_allocator<agg24::rgba8> span_alloc;

			// 1. Switch on filter type.
            switch (img.get_image_interpolation())
            {
                case nearest:
                {
                    typedef typename kiva::image_filters<other_format>::nearest_type span_gen_type;
                    typedef typename kiva::image_filters<other_format>::source_type source_type;

                    source_type source(src_pix, back_color);
                    span_gen_type span_generator(source, interpolator);
                    this->transform_image_final(img_outline, span_generator);
					break;
                }
                case bilinear:
                {
                    typedef typename kiva::image_filters<other_format>::bilinear_type span_gen_type;
                    typedef typename kiva::image_filters<other_format>::source_type source_type;

                    source_type source(src_pix, back_color);
                    span_gen_type span_generator(source, interpolator);
                    this->transform_image_final(img_outline, span_generator);
                    break;
                }
                case bicubic:
                case spline16:
                case spline36:
                case sinc64:
                case sinc144:
                case sinc256:
                case blackman64:
                case blackman100:
                case blackman256:
                {
                    agg24::image_filter_lut filter;
                    switch (img.get_image_interpolation())
                    {
                        case bicubic:
                            filter.calculate(agg24::image_filter_bicubic());
                            break;
                        case spline16:
                            filter.calculate(agg24::image_filter_spline16());
                            break;
                        case spline36:
                            filter.calculate(agg24::image_filter_spline36());
                            break;
                        case sinc64:
                            filter.calculate(agg24::image_filter_sinc64());
                            break;
                        case sinc144:
                            filter.calculate(agg24::image_filter_sinc144());
                            break;
                        case sinc256:
                            filter.calculate(agg24::image_filter_sinc256());
                            break;
                        case blackman64:
                            filter.calculate(agg24::image_filter_blackman64());
                            break;
                        case blackman100:
                            filter.calculate(agg24::image_filter_blackman100());
                            break;
                        case blackman256:
                            filter.calculate(agg24::image_filter_blackman256());
                            break;

                        case nearest:
                        case bilinear:
                            break;
                    }

                    typedef typename kiva::image_filters<other_format>::general_type span_gen_type;
                    typedef typename kiva::image_filters<other_format>::source_type source_type;

                    source_type source(src_pix, back_color);
                    span_gen_type span_generator(source, interpolator, filter);
                    this->transform_image_final(img_outline, span_generator);

                    break;
                }

            }

        }


        private:
        template<class span_gen_type>
        void transform_image_final(agg24::path_storage& img_outline,
                                   span_gen_type span_generator)
        {

            typedef agg24::span_allocator<agg24::rgba8> span_alloc_type;
            span_alloc_type span_allocator;
            agg24::scanline_u8 scanline;
   			agg24::rasterizer_scanline_aa<> rasterizer;

            if (this->state.alpha != 1.0)
            {
                rasterizer.gamma(alpha_gamma(this->state.alpha, 1.0));
            }

 			// fix me: This isn't handling clipping. [ Test. I think it should now]
            rasterizer.add_path(img_outline);
            agg24::render_scanlines_aa(rasterizer, scanline, this->renderer,
                                     span_allocator, span_generator);

       }
    };

    template <class agg_pixfmt>
    graphics_context<agg_pixfmt>::graphics_context(unsigned char *data,
                     int width, int height, int stride,
                     kiva::interpolation_e interp):
                     graphics_context_base(data,width,height,stride,interp),
                     renderer_pixfmt(buf),
                     //renderer_single_clip(renderer_pixfmt),
                     renderer(renderer_pixfmt)
    {
       // Required to set the clipping area of the renderer to the size of the buf.
       this->clear_clip_path();
    }

    template <class agg_pixfmt>
    kiva::pix_format_e graphics_context<agg_pixfmt>::format()
    {
        // The following dummy parameter is needed to pass in to agg_pix_to_kiva
        // because MSVC++ 6.0 doesn't properly handle template function
        // specialization (see notes in kiva_pix_format.h).
        agg_pixfmt *msvc6_dummy = NULL;
        return kiva::agg_pix_to_kiva(msvc6_dummy);
    }


    //---------------------------------------------------------------
    // Restore state
    //---------------------------------------------------------------

    template <class agg_pixfmt>
    void graphics_context<agg_pixfmt>::restore_state()
    {
        if (this->state_stack.size() == 0)
        {
            return;
        }

        this->state = this->state_stack.top();
        this->state_stack.pop();
        this->path.restore_ctm();

        // clear clippings paths and make renderer visible
        if (this->state.use_rect_clipping())
        {
            if (this->state.device_space_clip_rects.size() > 0)
            {
                this->renderer.reset_clipping(true);

                // add all the clipping rectangles in sequence
                std::vector<kiva::rect_type>::iterator it;
                for (it = this->state.device_space_clip_rects.begin();
                     it < this->state.device_space_clip_rects.end(); it++)
                {
                    this->renderer.add_clip_box(int(it->x), int(it->y), int(it->x2()), int(it->y2()));
                }
            }
            else
            {
                this->renderer.reset_clipping(false);
            }
        }
        else
        {
        	this->renderer.reset_clipping(true);
        	this->state.clipping_path = this->path;
        }
    }

    //---------------------------------------------------------------
    // Clipping path manipulation
    //---------------------------------------------------------------

    template <class agg_pixfmt>
    void graphics_context<agg_pixfmt>::clip()
    {
//    	this->state.clipping_path = this->path;

        agg24::scanline_p8 scanline;

        agg24::renderer_scanline_aa_solid< renderer_base_type >
                    aa_renderer(this->renderer);

        agg24::rgba transparent = this->state.fill_color;
        transparent.a = 0;

        aa_renderer.color(transparent);

        this->stroke_path_scanline_aa(this->state.clipping_path,
                                      aa_renderer, scanline);
    }

    template <class agg_pixfmt>
    void graphics_context<agg_pixfmt>::even_odd_clip()
    {
        throw kiva::even_odd_clip_error;
    }

    template <class agg_pixfmt>
    kiva::rect_type graphics_context<agg_pixfmt>::transform_clip_rectangle(const kiva::rect_type &rect)
    {

        // This only works if the ctm doesn't have any rotation.
        // otherwise, we need to use a clipping path. Test for this.
        agg24::trans_affine tmp(this->path.get_ctm());
        if ( !only_scale_and_translation(tmp))
        {
            throw kiva::ctm_rotation_error;
        }

        double x = rect.x;
        double y = rect.y;
        double x2 = rect.x2();
        double y2 = rect.y2();
        this->path.get_ctm().transform(&x, &y);
        this->path.get_ctm().transform(&x2, &y2);

        // fix me: How should we round here?
        // maybe we should lrint, but I don't think it is portable.  See
        // here: http://www.cs.unc.edu/~sud/tips/Programming_Tips.html
        x = int(floor(x+0.5));
        y = int(floor(y+0.5));

        // subtract 1 to account for agg (inclusive) vs. kiva (exclusive) clipping
        x2 = int(floor(x2+0.5))-1;
        y2 = int(floor(y2+0.5))-1;
        //x2 = int(floor(x2+0.5));
        //y2 = int(floor(y2+0.5));

        return kiva::rect_type(x, y, x2-x, y2-y);
    }

    template <class agg_pixfmt>
    void graphics_context<agg_pixfmt>::clip_to_rect(double x, double y, double sx, double sy)
    {
        kiva::rect_type tmp(x, y, sx, sy);
        this->clip_to_rect(tmp);
    }

    template <class agg_pixfmt>
    void graphics_context<agg_pixfmt>::clip_to_rect(kiva::rect_type &rect)
    {
        // Intersect the input rectangle with the current clipping path.
        //
        // 2/3/2005 Robert Kern noted that the Mac version forces a clear
        // of the path when calling clip_to_rect.  We'll do the same to
        // lessen the potential for inconsistencies.
        this->path.remove_all();

        if (this->state.use_rect_clipping())
        {
            kiva::rect_type device_rect(transform_clip_rectangle(rect));

            // optimize for case when there is only one existing rectangle
            if (this->state.device_space_clip_rects.size() == 1)
            {
                kiva::rect_type old(this->state.device_space_clip_rects.back());
                this->state.device_space_clip_rects.pop_back();
                kiva::rect_type newrect(kiva::disjoint_intersect(old, device_rect));

                if ((newrect.w < 0) || (newrect.h < 0))
                {
                    // new clip rectangle doesn't intersect anything, so we push on
                    // an empty rect as the new clipping region.
                    this->renderer.reset_clipping(false);
                    this->state.device_space_clip_rects.push_back(kiva::rect_type(0, 0, -1, -1));
                }
                else
                {
                    this->renderer.reset_clipping(true);
                    this->renderer.add_clip_box(int(newrect.x), int(newrect.y),
                                                int(newrect.x2()), int(newrect.y2()));
                    this->state.device_space_clip_rects.push_back(newrect);
                }
            }
            else
            {
                // we need to compute the intersection of the new rectangle with
                // the current set of clip rectangles.  we assume that the existing
                // clip_rects are a disjoint set.
                this->state.device_space_clip_rects = kiva::disjoint_intersect(
                    this->state.device_space_clip_rects, device_rect);

                if (this->state.device_space_clip_rects.size() == 0)
                {
                    this->renderer.reset_clipping(false);
                    this->state.device_space_clip_rects.push_back(kiva::rect_type(0, 0, -1, -1));
                }
                else
                {
                    this->renderer.reset_clipping(true);
                    for (unsigned int i=0; i<this->state.device_space_clip_rects.size(); i++)
                    {
                        kiva::rect_type *tmp = &this->state.device_space_clip_rects[i];
                        this->renderer.add_clip_box(int(tmp->x), int(tmp->y),
                                                    int(tmp->x2()), int(tmp->y2()));
                    }
                }
            }
        }
        else
        {
            // We don't support non-rect clipping.
            throw clipping_path_unsupported;
        }
    }

    template <class agg_pixfmt>
    void graphics_context<agg_pixfmt>::clip_to_rects(kiva::rect_list_type &rects)
    {
        // calculate the disjoint union of the input rectangles
        kiva::rect_list_type new_rects = disjoint_union(rects);

        if (this->state.use_rect_clipping())
        {
            // tranform and clip each new rectangle against the current clip_rects
            kiva::rect_list_type result_rects;
            for (kiva::rect_iterator it = new_rects.begin(); it != new_rects.end(); it++)
            {
                kiva::rect_type device_rect(transform_clip_rectangle(*it));
                kiva::rect_list_type new_result_rects(
                    kiva::disjoint_intersect(this->state.device_space_clip_rects, device_rect));

                for (kiva::rect_iterator tmp_iter = new_result_rects.begin();
                     tmp_iter != new_result_rects.end();
                     tmp_iter++)
                {
                    result_rects.push_back(*tmp_iter);
                }
            }

            if (result_rects.size() == 0)
            {
                // All areas are clipped out.
                this->state.device_space_clip_rects.clear();
                this->state.device_space_clip_rects.push_back(kiva::rect_type(0, 0, -1, -1));
                this->renderer.reset_clipping(false);
            }
            else
            {
                // Reset the renderer's clipping and add each new clip rectangle
                this->renderer.reset_clipping(true);
                for (kiva::rect_iterator it2 = result_rects.begin();
                     it2 != result_rects.end(); it2++)
                {
                    this->renderer.add_clip_box(int(it2->x), int(it2->y),
                                                int(it2->x2()), int(it2->y2()));
                }
                this->state.device_space_clip_rects = result_rects;
            }
        }
        else
        {
            // We don't support non-rect clipping.
            throw clipping_path_unsupported;
        }
    }


    template <class agg_pixfmt>
    void graphics_context<agg_pixfmt>::clip_to_rects(double* new_rects, int Nrects)
    {
        kiva::rect_list_type rectlist;
        for (int rectNum=0; rectNum < Nrects; rectNum++)
        {
            int ndx = rectNum*4;
            rectlist.push_back(kiva::rect_type(new_rects[ndx], new_rects[ndx+1],
                               new_rects[ndx+2], new_rects[ndx+3]));
        }
        clip_to_rects(rectlist);
    }

    template <class agg_pixfmt>
    void graphics_context<agg_pixfmt>::clear_clip_path()
    {
        // clear the existing clipping paths
        this->state.clipping_path.remove_all();
        this->state.device_space_clip_rects.clear();

        // set everything visible again.
        this->renderer.reset_clipping(1);

        // store the new clipping rectangle back into the first
        // rectangle of the graphics state clipping rects.
        this->state.device_space_clip_rects.push_back(this->renderer.clip_box());
    }

    template <class agg_pixfmt>
    int graphics_context<agg_pixfmt>::get_num_clip_regions()
    {
        return this->state.device_space_clip_rects.size();
    }

    template <class agg_pixfmt>
    kiva::rect_type graphics_context<agg_pixfmt>::get_clip_region(unsigned int i)
    {
        if (i >= this->state.device_space_clip_rects.size())
        {
            return kiva::rect_type();
        }
        else
        {
            return this->state.device_space_clip_rects[i];
        }
    }


    //---------------------------------------------------------------
    // Painting paths (drawing and filling contours)
    //---------------------------------------------------------------

    template <class agg_pixfmt>
    void graphics_context<agg_pixfmt>::clear(agg24::rgba value)
    {
        this->renderer.clear(value);
    }

    /*
    template <class agg_pixfmt>
    void graphics_context<agg_pixfmt>::clear(double value)
    {
        this->renderer_single_clip.clear(value);
    }
    */

    template <class agg_pixfmt>
    void graphics_context<agg_pixfmt>::draw_path(draw_mode_e mode)
    {
        switch(mode)
        {
            case FILL:
                this->_fill_path(agg24::fill_non_zero);
                break;
            case EOF_FILL:
                this->_fill_path(agg24::fill_even_odd);
                break;
            case STROKE:
                this->_stroke_path();
                break;
            case FILL_STROKE:
                this->_fill_path(agg24::fill_non_zero);
                this->_stroke_path();
                break;
            case EOF_FILL_STROKE:
                this->_fill_path(agg24::fill_even_odd);
                this->_stroke_path();
                break;
        }
        this->path.remove_all();
    }

    template <class agg_pixfmt>
    void graphics_context<agg_pixfmt>::draw_rect(double rect[4],
                   draw_mode_e mode)
    {
        // Try a fast renderer first.
        int fast_worked = this->_draw_rect_simple(rect, mode);

        if (!fast_worked)
        {
            double x = rect[0];
            double y = rect[1];
            double sx = rect[2];
            double sy = rect[3];

            this->begin_path();
            this->move_to(x, y);
            this->line_to(x+sx, y);
            this->line_to(x+sx, y+sy);
            this->line_to(x, y+sy);
            this->close_path();
            this->draw_path(mode);
        }
        else
        {
            //printf("simple worked!\n");
        }

        this->path.remove_all();
    }

    template <class agg_pixfmt>
    int graphics_context<agg_pixfmt>::_draw_rect_simple(double rect[4],
                                                        draw_mode_e mode)
    {
        /* function requires that antialiasing is false and ctm doesn't
           have any rotation.
        */
        //printf("trying _simple\n");
        int success = 0;
        agg24::trans_affine ctm = this->get_ctm();

        if ( !this->state.should_antialias &&
              only_scale_and_translation(ctm) &&
             (this->state.line_width == 1.0 ||
              this->state.line_width == 0.0))
             // fix me: should test for join style
             //&& this->state.line_join == JOIN_MITER )
        {
            agg24::renderer_primitives<renderer_base_type>
                                       renderer(this->renderer);

            renderer.fill_color(this->get_fill_color());
            // use transparency to indicate a 0 width line
            agg24::rgba line_color = this->get_stroke_color();
            line_color.a *= this->state.line_width;
            renderer.line_color(line_color);

            double temp[6];
            ctm.store_to(temp);
            double scale_x = temp[0];
            double scale_y = temp[3];
            double tx = temp[4];
            double ty = temp[5];

            //printf("rect: %d, %d %d, %d\n", rect[0], rect[1], rect[2], rect[3]);
            //printf("trans, scale: %d, %d %d, %d\n", tx, ty, scale_x, scale_y);
            // fix me: need to handle rounding here...
            int x1 = int(rect[0]*scale_x + tx);
            int y1 = int(rect[1]*scale_y + ty);
            int x2 = int((rect[0]+rect[2])*scale_x + tx);
            int y2 = int((rect[1]+rect[3])*scale_y + ty);


            if (mode == FILL_STROKE ||
                mode == EOF_FILL_STROKE)
            {
                //printf("fill stroke: %d, %d %d, %d\n", x1, y1, x2, y2);
                renderer.outlined_rectangle(x1, y1, x2, y2);
                // This isn't right, but it should be faster.  Interestingly,
                // it didn't seem to be.
                //this->renderer.copy_bar(x1, y1, x2, y2, this->get_fill_color());
                success = 1;
            }
            else if (mode == STROKE )
            {
                //printf("stroke: %d, %d %d, %d\n", x1, y1, x2, y2);
                renderer.rectangle(x1, y1, x2, y2);
                success = 1;
            }
            else if (mode == FILL ||
                     mode == EOF_FILL )
            {
                //printf("fill: %d, %d %d, %d\n", x1, y1, x2, y2);
                renderer.solid_rectangle(x1, y1, x2, y2);
                success = 1;
            }
        }

        return success;
    }


    template <class agg_pixfmt>
    int graphics_context<agg_pixfmt>::draw_marker_at_points(double* pts,int Npts,int size,
                               agg24::marker_e type)
    {
        int success = 0;
        agg24::trans_affine ctm = this->get_ctm();

        if ( only_scale_and_translation(ctm) &&
            (this->state.line_width == 1.0 ||
             this->state.line_width == 0.0))
             //&& this->state.line_join == JOIN_MITER )
        {
            // TODO-PZW: fix this!!
            agg24::renderer_markers< renderer_base_type >
                                       m(this->renderer);
            m.fill_color(this->get_fill_color());
            // use transparency to indicate an 0 width line
            agg24::rgba line_color = this->get_stroke_color();
            line_color.a *= this->state.line_width;
            m.line_color(line_color);

            double mx, my, sx, sy;
            get_scale(ctm, &sx, &sy);

            for(int i = 0; i < Npts*2; i+=2)
            {
                mx = pts[i];
                my = pts[i+1];
                ctm.transform(&mx, &my);
                // XXX: Assuming scale is uniform in both directions
                m.marker(int(mx), int(my), size * sx, type);
            }
            success = 1;
        }
        return success;
    }

    template <class agg_pixfmt>
    void graphics_context<agg_pixfmt>::draw_path_at_points(double* pts,int Npts,
                              kiva::compiled_path& marker,
                              draw_mode_e mode)
    {
        // This routine draws a path (i.e. marker) at multiple points
        // on the screen.  It is used heavily when rendering scatter
        // plots.
        //
        // The routine has been special cased to handle the filling
        // markers without outlining them when the ctm doesn't have
        // rotation, scaling, or skew.
        //
        // fastest path
        //     (1) We're only using FILL mode.
        //     (2) ctm is identity matrix
        // fast path
        //     (1) We're only using FILL mode.
        //     (2) ctm just has translational components (tx,ty)
        // normal path
        //     Everything else.


        // !! This path commented out.  We don't have any cases
        // !! currently where draw_marker_at_points won't work,
        // !! so we provide fast and slow without taking the time
        // !! to update the inbetween route.
        // no outline

        if (0) //(!(mode & STROKE) || state.line_color.a == 0.0)
        {
            /*
            // set fill color -- multiply by alpha if it is set.
            agg24::rgba color;
            color = this->state.fill_color;
            color.a *= this->state.alpha;
            this->renderer.attribute(color);

            agg24::trans_affine ctm = this->get_ctm();
            // set the rasterizer filling rule
            if (mode & FILL)
                this->rasterizer.filling_rule(agg24::fill_non_zero);
            else if (mode & EOF_FILL)
                this->rasterizer.filling_rule(agg24::fill_even_odd);

            // fastest path
            if (is_identity(ctm))
            {
                for(int i = 0; i < Npts*2; i+=2)
                {
                    const double x = pts[i];
                    const double y = pts[i+1];
                    this->rasterizer.add_path(marker,x,y);
                    this->rasterizer.render(renderer);
                }
            }
            // 2nd fastest path
            else if (only_translation(ctm))
            {
                double temp[6];
                this->get_ctm().store_to(temp);
                double tx = temp[4];
                double ty = temp[5];
                for(int i = 0; i < Npts*2; i+=2)
                {
                    const double x = pts[i] + tx;
                    const double y = pts[i+1] + ty;
                    this->rasterizer.add_path(marker,x,y);
                    this->rasterizer.render(renderer);
                }
            }
            */
        }
        // outlined draw mode or
        // complicated ctm (rotation,scaling, or skew)
        else
        {
            this->begin_path();
            for(int i = 0; i < Npts*2; i+=2)
            {
                const double x = pts[i];
                const double y = pts[i+1];
                // This is faster than saving the entire state.
                this->path.save_ctm();
                this->translate_ctm(x,y);
                this->add_path(marker);
                this->draw_path(mode);
                this->path.restore_ctm();
            }
        }

    }

    template <class agg_pixfmt>
    bool graphics_context<agg_pixfmt>::show_text(char*text)
    {
        typedef agg24::glyph_raster_bin<agg24::rgba8>                   GlyphGeneratorType;
        typedef agg24::renderer_scanline_aa_solid<renderer_base_type> ScanlineRendererType;

        //GlyphGeneratorType glyphGen(0);
        ScanlineRendererType scanlineRenderer(this->renderer);

        const agg24::glyph_cache *glyph = NULL;

        // Explicitly decode UTF8 bytes to 32-bit codepoints to feed into the
        // font API.
        size_t text_length = strlen(text);
        utf8::iterator<char*> p(text, text, text+text_length);
        utf8::iterator<char*> p_end(text+text_length, text, text+text_length);

        bool retval = true;

        // Check to make sure the font's loaded.
        if (!this->is_font_initialized())
        {
            return false;
        }

        this->_grab_font_manager();
        font_engine_type *font_engine = kiva::GlobalFontEngine();
        font_manager_type *font_manager = kiva::GlobalFontManager();

        // Concatenate the CTM with the text matrix to get the full transform for the
        // font engine.
    	agg24::trans_affine full_text_xform(this->text_matrix * this->path.get_ctm());

       // the AGG freetype transform is a per character transform.  We need to remove the
       // offset part of the transform to prevent that offset from occuring between each
       // character.  We'll handle the intial offset ourselves.
       double start_x, start_y;
       double text_xform_array[6];
       full_text_xform.store_to(text_xform_array);
       // Pull the translation values out of the matrix as our starting offset and
       // then replace them with zeros for use in the font engine.
       start_x = text_xform_array[4];
       start_y = text_xform_array[5];

       text_xform_array[4] = 0.0;
       text_xform_array[5] = 0.0;

       full_text_xform.load_from(text_xform_array);
       font_engine->transform(full_text_xform);

        if (this->state.text_drawing_mode == kiva::TEXT_FILL)
        {
            scanlineRenderer.color(this->state.fill_color);
        }
        else if ((this->state.text_drawing_mode == kiva::TEXT_STROKE) ||
                 (this->state.text_drawing_mode == kiva::TEXT_FILL_STROKE))
        {
            scanlineRenderer.color(this->state.line_color);
        }

        double advance_x = 0.0;
        double advance_y = 0.0;

        for (; p!=p_end; ++p)
        {
            double x = start_x + advance_x;
            double y = start_y + advance_y;
            glyph = font_manager->glyph(*p);

            if (glyph == NULL)
            {
                retval = false;
                break;
            }
            font_manager->add_kerning(&x, &y);
            font_manager->init_embedded_adaptors(glyph, x, y);
            if (this->state.text_drawing_mode != kiva::TEXT_INVISIBLE)
            {
                agg24::render_scanlines(font_manager->gray8_adaptor(),
                                      font_manager->gray8_scanline(),
                                      scanlineRenderer);
            }

            advance_x += glyph->advance_x;
            advance_y += glyph->advance_y;
        }

        agg24::trans_affine null_xform = agg24::trans_affine_translation(0., 0.);
        font_engine->transform(null_xform);
        this->_release_font_manager();

        agg24::trans_affine trans = agg24::trans_affine_translation(advance_x,
    	                                                        advance_y);
        this->text_matrix.multiply(trans);
        return retval;
    }

    template <class agg_pixfmt>
    int graphics_context<agg_pixfmt>::draw_image(kiva::graphics_context_base* img,
                                                 double rect[4], bool force_copy)
    {
        int success = 0;

        // We have to scale first and then translate; otherwise, Agg will cause
        // the translation to be scaled as well.
        double sx = rect[2]/img->width();
        double sy = rect[3]/img->height();
        agg24::trans_affine img_mtx = agg24::trans_affine_scaling(sx,sy);

        img_mtx *= agg24::trans_affine_translation(rect[0],rect[1]);

        img_mtx *= this->path.get_ctm();

        double tx, ty;
        get_translation(img_mtx, &tx, &ty);


        //success = transform_image(img, img_mtx);


        // The following section attempts to use a fast method for blending in
        // cases where the full interpolation methods aren't needed.

        // When there isn't any scaling or rotation, try a fast method for
        // copy or blending the pixels.  They will fail if pixel formats differ...
        // If the user is forcing us to respect the blend_copy mode regardless
        // of the CTM, then we make it so.
        // fix me: Not testing whether tx, ty are (nearly) integer values.
        //        We should.
        if (only_translation(img_mtx) || force_copy)
        {
            if (this->state.blend_mode == kiva::blend_copy)
            {
                success = this->copy_image(img, (int) tx, (int) ty);
            }
            else
            {
                success = this->blend_image(img, (int)tx, (int)ty);
            }
        }

        if (!success)
        {
            // looks like the fast approach didn't work -- there is some
            // transform to the matrix so we'll use an interpolation scheme.

            // We're just starting blend_mode support.  From here down, we
            // only support normal.
            if (!(this->state.blend_mode == kiva::blend_normal))
            {
                success = 0;
                return success;
            }

            success = transform_image(img, img_mtx);

        }

        return success;
    }

    template <class agg_pixfmt>
    int graphics_context<agg_pixfmt>::copy_image(kiva::graphics_context_base* img,
                                                 int tx, int ty)
    {
        // This function is only valid if only_translation(ctm) == True and
        // image is not to be scaled.

        int success = 0;

        // Copy only works if images have the same format.
        // fix me: This restriction should be fixed.
        // fix me: We are ignoring that tx and ty are double.  test that
        //         we are close.  Otherwise, we need to use interpolation.
        if (img->format() != this->format())
        {
            //doesn't meet requirements
			printf("copy_image() on this gc requires format %d, got %d.",
				   this->format(), img->format());
            success = 0;
        }
        else
        {
            agg24::rect_i r(0, 0, img->width(), img->height());
            this->renderer.copy_from(img->buf, &r, tx, ty);
            success = 1;
        }
        return success;
    }

    template <class agg_pixfmt>
    int graphics_context<agg_pixfmt>::blend_image(kiva::graphics_context_base* img,
                                                  int tx, int ty)
    {
        // This function is only valid if only_translation(ctm) == True and
        // image is not to be scaled.
        // Note: I thought I needed to negate the tx,ty in here, but it doesn't
        // turn out to be true.

        int success = 0;
        unsigned int alpha = unsigned(this->state.alpha*255);

        // Check that format match.  I think the formats
        // actually only have to have the same number of channels,
        // so this test is to restrictive.
        // fix me: lighten up this format restrictions.
        // fix me: We are ignoring that tx and ty are double.  test that
        //         we are close.  Otherwise, we need to use interpolation.
        if (img->format() != this->format())
        {
            //doesn't meet requirements
            success = 0;
        }
        else
        {
            agg24::rect_i r(0, 0, img->width(), img->height());

            switch (img->format())
            {
                // fix me: agg 2.4 doesn't work for blending rgb values into other buffers.
                //         I think this should be fixed, but I also think it would take
                //         some major agg hackery.
                case kiva::pix_format_rgb24:
                case kiva::pix_format_bgr24:
                    success = 0;
                    break;

                //case kiva::pix_format_rgb24:
                //{
                //    typedef kiva::graphics_context<agg24::pixfmt_rgb24> pix_format_type;
                //    this->renderer.blend_from(static_cast<pix_format_type* >(img)->renderer_pixfmt,
                //                              &r, tx, ty, alpha);
                //    success = 1;
                //    break;
                //}
                //
                //case kiva::pix_format_bgr24:
                //{
                //    typedef kiva::graphics_context<agg24::pixfmt_bgr24> pix_format_type;
                //    this->renderer.blend_from(static_cast<pix_format_type* >(img)->renderer_pixfmt,
                //                              &r, tx, ty, alpha);
                //    success = 1;
                //    break;
                //}

                case kiva::pix_format_rgba32:
                {
                    typedef kiva::graphics_context<agg24::pixfmt_rgba32> pix_format_type;
                    this->renderer.blend_from(static_cast<pix_format_type* >(img)->renderer_pixfmt,
                                              &r, tx, ty, alpha);
                    success = 1;
                    break;
                }
                case kiva::pix_format_argb32:
                {
                    typedef kiva::graphics_context<agg24::pixfmt_argb32> pix_format_type;
                    this->renderer.blend_from(static_cast<pix_format_type* >(img)->renderer_pixfmt,
                                              &r, tx, ty, alpha);
                    success = 1;
                    break;
                }
                case kiva::pix_format_abgr32:
                {
                    typedef kiva::graphics_context<agg24::pixfmt_abgr32> pix_format_type;
                    this->renderer.blend_from(static_cast<pix_format_type* >(img)->renderer_pixfmt,
                                              &r, tx, ty, alpha);
                    success = 1;
                    break;
                }
                case kiva::pix_format_bgra32:
                {
                    typedef kiva::graphics_context<agg24::pixfmt_bgra32> pix_format_type;
                    this->renderer.blend_from(static_cast<pix_format_type* >(img)->renderer_pixfmt,
                                              &r, tx, ty, alpha);
                    success = 1;
                    break;
                }
                case kiva::pix_format_undefined:
                case kiva::pix_format_gray8:
                case kiva::pix_format_rgb555:
                case kiva::pix_format_rgb565:
                case kiva::end_of_pix_formats:
                default:
                {
                    // format not valid.
                    success = 0;
                }
            }
        }

        return success;
    }

    template <class agg_pixfmt>
    int graphics_context<agg_pixfmt>::transform_image(kiva::graphics_context_base* img,
                                                      agg24::trans_affine& img_mtx)
    {
        int success = 0;

        switch (img->format())
        {
            case kiva::pix_format_rgb24:
            {
                typedef kiva::graphics_context<agg24::pixfmt_rgb24> gc_type;
                this->transform_image_interpolate(*(static_cast<gc_type*>(img)), img_mtx);
                success = 1;
                break;
            }
            case kiva::pix_format_bgr24:
            {

                typedef kiva::graphics_context<agg24::pixfmt_bgr24> gc_type;
                this->transform_image_interpolate(*(static_cast<gc_type*>(img)), img_mtx);
                success = 1;
                break;
            }
            case kiva::pix_format_rgba32:
            {
                typedef kiva::graphics_context<agg24::pixfmt_rgba32> gc_type;
                this->transform_image_interpolate(*(static_cast<gc_type*>(img)), img_mtx);
                success = 1;
                break;
            }
            case kiva::pix_format_argb32:
            {
                typedef kiva::graphics_context<agg24::pixfmt_argb32> gc_type;
                this->transform_image_interpolate(*(static_cast<gc_type*>(img)),img_mtx);
                success = 1;
                break;
            }
            case kiva::pix_format_abgr32:
            {
                typedef kiva::graphics_context<agg24::pixfmt_abgr32> gc_type;
                this->transform_image_interpolate(*(static_cast<gc_type*>(img)),img_mtx);
                success = 1;
                break;
            }
            case kiva::pix_format_bgra32:
            {
                typedef kiva::graphics_context<agg24::pixfmt_bgra32> gc_type;
                this->transform_image_interpolate(*(static_cast<gc_type*>(img)),img_mtx);
                success = 1;
                break;
            }
            case kiva::pix_format_undefined:
            case kiva::pix_format_gray8:
            case kiva::pix_format_rgb555:
            case kiva::pix_format_rgb565:
            case kiva::end_of_pix_formats:
            default:
            {
                // format not valid.
                success = 0;
            }
        }

        return success;
    }

    typedef graphics_context<agg24::pixfmt_rgb24> graphics_context_rgb24;
    typedef graphics_context<agg24::pixfmt_bgr24> graphics_context_bgr24;
    typedef graphics_context<agg24::pixfmt_bgra32> graphics_context_bgra32;
    typedef graphics_context<agg24::pixfmt_rgba32> graphics_context_rgba32;
    typedef graphics_context<agg24::pixfmt_argb32> graphics_context_argb32;
    typedef graphics_context<agg24::pixfmt_abgr32> graphics_context_abgr32;

}  // namespace kiva

#endif
