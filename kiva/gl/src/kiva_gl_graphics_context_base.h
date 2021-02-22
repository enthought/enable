// (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
// All rights reserved.
//
// This software is provided without warranty under the terms of the BSD
// license included in LICENSE.txt and may be redistributed only under
// the conditions described in the aforementioned license. The license
// is also available online at http://www.enthought.com/licenses/BSD.txt
//
// Thanks for using Enthought open source!
#ifndef KIVA_GL_GRAPHICS_CONTEXT_BASE_H
#define KIVA_GL_GRAPHICS_CONTEXT_BASE_H

#include <stack>
#include <vector>

#include "agg_basics.h"
#include "agg_color_rgba.h"

#include "kiva_gl_constants.h"
#include "kiva_gl_rect.h"
#include "kiva_gl_graphics_state.h"
#include "kiva_gl_affine_helpers.h"

namespace kiva_gl
{
    class graphics_context_base
    {
    public:
        // The current path.  This also includes the ctm.
        kiva_gl::compiled_path path;

        kiva_gl::graphics_state state;
        std::stack<kiva_gl::graphics_state> state_stack;

        // fix me: Not sure this should be here, but, putting it here completely
        //         unifies images and graphics contexts.
        // (TODO-PZW: revisit this)
        kiva_gl::interpolation_e _image_interpolation;

        graphics_context_base(kiva_gl::interpolation_e interp);
        virtual ~graphics_context_base();

        int width();
        int height();
        int stride();
        int bottom_up();

        virtual kiva_gl::pix_format_e format() = 0;

        kiva_gl::interpolation_e get_image_interpolation();
        void set_image_interpolation(interpolation_e interpolation);

        //---------------------------------------------------------------
        // set graphics_state values
        //---------------------------------------------------------------

        void set_stroke_color(kiva_gl_agg::rgba& value);
        kiva_gl_agg::rgba& get_stroke_color();

        // TODO-PZW: do we need corresponding get() functions for
        // all of the following?

        void set_line_width(double value);
        void set_line_join(line_join_e value);
        void set_line_cap(line_cap_e value);
        void set_line_dash(double* pattern, int n, double phase=0);

        // fix me: Blend mode is *barely* supported and
        //         probably abused (my copy setting).
        void set_blend_mode(blend_mode_e value);
        kiva_gl::blend_mode_e get_blend_mode();

        void set_fill_color(kiva_gl_agg::rgba& value);

        // need get method for freetype renderer.
        // should I return a reference??
        kiva_gl_agg::rgba& get_fill_color();

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
        void concat_ctm(kiva_gl_agg::trans_affine& m);
        void set_ctm(kiva_gl_agg::trans_affine& m);
        kiva_gl_agg::trans_affine get_ctm();

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
        void add_path(kiva_gl::compiled_path& other_path);
        compiled_path _get_path();
        kiva_gl::rect_type _get_path_bounds();

        void lines(double* pts, int Npts);
        void line_set(double* start, int Nstart, double* end, int Nend);

        void rect(double x, double y, double sx, double sy);
        void rect(kiva_gl::rect_type &rect);
        void rects(double* all_rects, int Nrects);
        void rects(kiva_gl::rect_list_type &rectlist);

        kiva_gl_agg::path_storage boundary_path(kiva_gl_agg::trans_affine& affine_mtx);

        //---------------------------------------------------------------
        // Clipping path manipulation
        //---------------------------------------------------------------
        virtual void clip() = 0;
        virtual void even_odd_clip() = 0;
        virtual void clip_to_rect(double x, double y, double sx, double sy) = 0;
        virtual void clip_to_rect(kiva_gl::rect_type &rect) = 0;
        virtual void clip_to_rects(double* new_rects, int Nrects) = 0;
        virtual void clip_to_rects(kiva_gl::rect_list_type &rects) = 0;
        virtual void clear_clip_path() = 0;

        // The following two are meant for debugging purposes, and are not part
        // of the formal interface for GraphicsContexts.
        virtual int get_num_clip_regions() = 0;
        virtual kiva_gl::rect_type get_clip_region(unsigned int i) = 0;

        //---------------------------------------------------------------
        // Painting paths (drawing and filling contours)
        //---------------------------------------------------------------
        virtual void clear(kiva_gl_agg::rgba value=kiva_gl_agg::rgba(1, 1, 1, 1)) = 0;

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
                                          kiva_gl::marker_e type=kiva_gl::marker_square) = 0;

        virtual void draw_path_at_points(double* pts,int Npts,
                                         kiva_gl::compiled_path& marker,
                                         draw_mode_e mode) = 0;

        //---------------------------------------------------------------
        // Image handling
        //---------------------------------------------------------------

        // Draws an image into the rectangle specified as (x, y, width, height);
        // The image is scaled and/or stretched to fit inside the rectangle area
        // specified.
        virtual int draw_image(kiva_gl::graphics_context_base* img, double rect[4], bool force_copy=false) = 0;
        int draw_image(kiva_gl::graphics_context_base* img);

    };

}

#endif /* KIVA_GL_GRAPHICS_CONTEXT_BASE_H */
