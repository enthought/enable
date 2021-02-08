// (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
// All rights reserved.
//
// This software is provided without warranty under the terms of the BSD
// license included in LICENSE.txt and may be redistributed only under
// the conditions described in the aforementioned license. The license
// is also available online at http://www.enthought.com/licenses/BSD.txt
//
// Thanks for using Enthought open source!
#ifndef KIVA_GRAPHICS_CONTEXT_BASE_H
#define KIVA_GRAPHICS_CONTEXT_BASE_H

#define KIVA_USE_FREETYPE
#ifdef KIVA_USE_FREETYPE
#include "agg_font_freetype.h"
#endif
#ifdef KIVA_USE_WIN32
#include "agg_font_win32_tt.h"
#endif


#include <stack>
#include <vector>

#include "agg_basics.h"
#include "agg_color_rgba.h"
#include "agg_rendering_buffer.h"
#include "agg_renderer_markers.h"

#include "kiva_constants.h"
#include "kiva_pix_format.h"
#include "kiva_rect.h"
#include "kiva_graphics_state.h"
#include "kiva_affine_helpers.h"

// text includes
#include "agg_glyph_raster_bin.h"
#include "agg_renderer_scanline.h"
#include "agg_renderer_raster_text.h"
#include "agg_embedded_raster_fonts.h"

#include "agg_font_cache_manager.h"



namespace kiva
{

#ifdef KIVA_USE_FREETYPE
    typedef agg24::font_engine_freetype_int32 font_engine_type;
#endif
#ifdef KIVA_USE_WIN32
    typedef agg24::font_engine_win32_tt_int32 font_engine_type;
#endif
    typedef agg24::font_cache_manager<font_engine_type> font_manager_type;

    font_engine_type* GlobalFontEngine();
    font_manager_type* GlobalFontManager();
    void cleanup_font_threading_primitives();

	class graphics_context_base
	{
	public:
        // The current path.  This also includes the ctm.
        kiva::compiled_path path;

        // The text matrix is *not* part of the graphics state.
        agg24::trans_affine text_matrix;

        kiva::graphics_state state;
        std::stack<kiva::graphics_state> state_stack;

        agg24::rendering_buffer buf;

        // fix me: Not sure this should be here, but, putting it here completely
        //         unifies images and graphics contexts.
        // (TODO-PZW: revisit this)
        kiva::interpolation_e _image_interpolation;

        // text handling.

        graphics_context_base(unsigned char *data,
                 int width, int height, int stride,
                 kiva::interpolation_e interp);

        virtual ~graphics_context_base();

        int width();
        int height();
        int stride();
        int bottom_up();

		virtual kiva::pix_format_e format() = 0;

        agg24::rendering_buffer& rendering_buffer();
        kiva::interpolation_e get_image_interpolation();
        void set_image_interpolation(interpolation_e interpolation);

        //---------------------------------------------------------------
        // set graphics_state values
        //---------------------------------------------------------------

        void set_stroke_color(agg24::rgba& value);
        agg24::rgba& get_stroke_color();

        // TODO-PZW: do we need corresponding get() functions for
        // all of the following?

        void set_line_width(double value);
        void set_line_join(line_join_e value);
        void set_line_cap(line_cap_e value);
        void set_line_dash(double* pattern, int n, double phase=0);

        // fix me: Blend mode is *barely* supported and
        //         probably abused (my copy setting).
        void set_blend_mode(blend_mode_e value);
        kiva::blend_mode_e get_blend_mode();

        void set_fill_color(agg24::rgba& value);

        // need get method for freetype renderer.
        // should I return a reference??
        agg24::rgba& get_fill_color();

        // need get method for freetype renderer.
        // fix me: Is the get method still needed?
        void set_alpha(double value);
        double get_alpha();

        // need get method for freetype renderer.
        // fix me: Is the get method still needed?
        void set_antialias(int value);
        int get_antialias();

        // TODO-PZW: get() functions needed?
        void set_miter_limit(double value);
        void set_flatness(double value);

        //---------------------------------------------------------------
        // text and font functions
        //---------------------------------------------------------------

        // Text drawing information needs to have get/set methods
        void set_text_position(double tx, double ty);
        void get_text_position(double* tx, double* ty);

        void set_text_matrix(agg24::trans_affine& value);
        agg24::trans_affine get_text_matrix();

        void set_character_spacing(double value);
        double get_character_spacing();

        void set_text_drawing_mode(text_draw_mode_e value);

        // The following methods all return true if successful and false
        // otherwise.
        bool set_font(kiva::font_type& font);
        bool is_font_initialized();
        bool set_font_size(int size);
        virtual bool show_text(char *text)=0;

        bool show_text_at_point(char *text, double tx, double ty);

        // This will always return a font_type object.  The font's
        // is_loaded() method should be checked to see if the font is valid.
        kiva::font_type& get_font();

        // Returns a rectangle representing the bounds of the text string.
        // The rectangle is measured in the transformed space of the text
        // itself, and its origin is not necessarily 0,0 (for fonts with
        // overhanging glyphs).
        // is_font_initialized() should be checked to make sure the font
        // has been properly loaded and initialized.
        kiva::rect_type get_text_extent(char *text);

        bool get_text_bbox_as_rect(char *text);

        //---------------------------------------------------------------
        // save/restore graphics state
        //---------------------------------------------------------------

        void save_state();
        virtual void restore_state() = 0;

        //---------------------------------------------------------------
        // coordinate transform matrix transforms
        //---------------------------------------------------------------

        void translate_ctm(double x, double y);
        void rotate_ctm(double angle);
        void scale_ctm(double sx, double sy);
        void concat_ctm(agg24::trans_affine& m);
        void set_ctm(agg24::trans_affine& m);
        agg24::trans_affine get_ctm();
        void get_freetype_text_matrix(double* out);

        //---------------------------------------------------------------
        // Sending drawing data to a device
        //---------------------------------------------------------------

        void flush();
        void synchronize();

        //---------------------------------------------------------------
        // Page Definitions
        //---------------------------------------------------------------

        void begin_page();
        void end_page();

        //---------------------------------------------------------------
        // Path operations
        //---------------------------------------------------------------

        void begin_path();
        void move_to(double x, double y);
        void line_to( double x, double y);
        void curve_to(double cpx1, double cpy1,
                      double cpx2, double cpy2,
                      double x, double y);

        void quad_curve_to(double cpx, double cpy,
                           double x, double y);

        // arc() and arc_to() draw circular segments.  When the arc
        // is added to the current path, it may become an elliptical
        // arc depending on the CTM.

        // Draws a circular segment centered at the point (x,y) with the
        // given radius.
        void arc(double x, double y, double radius, double start_angle,
            double end_angle, bool cw=false);

        // Sweeps a circular arc from the pen position to a point on the
        // line from (x1,y1) to (x2,y2).
        // The arc is tangent to the line from the current pen position
        // to (x1,y1), and it is also tangent to the line from (x1,y1)
        // to (x2,y2).  (x1,y1) is the imaginary intersection point of
        // the two lines tangent to the arc at the current point and
        // at (x2,y2).
        // If the tangent point on the line from the current pen position
        // to (x1,y1) is not equal to the current pen position, a line is
        // drawn to it.  Depending on the supplied radius, the tangent
        // point on the line fron (x1,y1) to (x2,y2) may or may not be
        // (x2,y2).  In either case, the arc is drawn to the point of
        // tangency, which is also the new pen position.
        //
        // Consider the common case of rounding a rectangle's upper left
        // corner.  Let "r" be the radius of rounding.  Let the current
        // pen position be (x_left + r, y_top).  Then (x2,y2) would be
        // (x_left, y_top - radius), and (x1,y1) would be (x_left, y_top).
        void arc_to(double x1, double y1, double x2, double y2, double radius);

        void close_path();
        void add_path(kiva::compiled_path& other_path);
        compiled_path _get_path();
        kiva::rect_type _get_path_bounds();

        void lines(double* pts, int Npts);
        void line_set(double* start, int Nstart, double* end, int Nend);

        void rect(double x, double y, double sx, double sy);
        void rect(kiva::rect_type &rect);
        void rects(double* all_rects, int Nrects);
        void rects(kiva::rect_list_type &rectlist);

		agg24::path_storage boundary_path(agg24::trans_affine& affine_mtx);

        //---------------------------------------------------------------
        // Clipping path manipulation
        //---------------------------------------------------------------
        virtual void clip() = 0;
        virtual void even_odd_clip() = 0;
        virtual void clip_to_rect(double x, double y, double sx, double sy) = 0;
        virtual void clip_to_rect(kiva::rect_type &rect) = 0;
        virtual void clip_to_rects(double* new_rects, int Nrects) = 0;
        virtual void clip_to_rects(kiva::rect_list_type &rects) = 0;
        virtual void clear_clip_path() = 0;

		// The following two are meant for debugging purposes, and are not part
		// of the formal interface for GraphicsContexts.
        virtual int get_num_clip_regions() = 0;
        virtual kiva::rect_type get_clip_region(unsigned int i) = 0;

        //---------------------------------------------------------------
        // Painting paths (drawing and filling contours)
        //---------------------------------------------------------------
        virtual void clear(agg24::rgba value=agg24::rgba(1,1,1,1)) = 0;
        //virtual void clear(double alpha) = 0;

        virtual void fill_path() = 0;
        virtual void eof_fill_path() = 0;

        virtual void stroke_path() = 0;
        virtual void _stroke_path() = 0;

        virtual void draw_path(draw_mode_e mode=FILL_STROKE) = 0;
        virtual void draw_rect(double rect[4],
                               draw_mode_e mode=FILL_STROKE) = 0;

        // Draw a marker at all the points in the list.  This is a
        // very fast function that only works in special cases.
        // The succeeds if the line_width != 0.0 or 1.0, the line_join
        // is set to JOIN_MITER (!! NOT CURRENTLY ENFORCED), and the
        // ctm only has translational components.
        //
        // Typically this is called before trying the more general
        // draw_path_at_points() command.  It is typically 5-10 times
        // faster.
        //
        // Returns: int
        //          0 on failure
        //          1 on success
        virtual int draw_marker_at_points(double* pts,int Npts,int size,
                                   agg24::marker_e type=agg24::marker_square) = 0;

        virtual void draw_path_at_points(double* pts,int Npts,
                                  kiva::compiled_path& marker,
                                  draw_mode_e mode) = 0;

        //---------------------------------------------------------------
        // Image handling
        //---------------------------------------------------------------

        // Draws an image into the rectangle specified as (x, y, width, height);
        // The image is scaled and/or stretched to fit inside the rectangle area
        // specified.
        virtual int draw_image(kiva::graphics_context_base* img, double rect[4], bool force_copy=false) = 0;
        int draw_image(kiva::graphics_context_base* img);

        // fix me: This is a temporary fix to help with speed issues in draw image.
        //         WE SHOULD GET RID OF IT AND SUPPORT DRAWING MODES.
        //virtual int copy_image(kiva::graphics_context_base* img, int tx, int ty);


        //---------------------------------------------------------------------
        // Gradient support
        //---------------------------------------------------------------------

        //
        //
        //
        virtual void linear_gradient(double x1, double y1, double x2, double y2,
							std::vector<kiva::gradient_stop> stops,
                            const char* spread_method,
                            const char* units="userSpaceOnUse");

        //
        //
        //
        virtual void radial_gradient(double cx, double cy, double r,
                            double fx, double fy,
							std::vector<kiva::gradient_stop> stops,
                            const char* spread_method,
                            const char* units="userSpaceOnUse");

    protected:
        // Grabs and configure the font engine with the settings on our current
        // state's font object.
        void _grab_font_manager();
        void _release_font_manager();

        bool _is_font_initialized;

    };



}

#endif /* KIVA_GRAPHICS_CONTEXT_BASE_H */
