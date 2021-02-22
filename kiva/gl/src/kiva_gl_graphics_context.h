// (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
// All rights reserved.
//
// This software is provided without warranty under the terms of the BSD
// license included in LICENSE.txt and may be redistributed only under
// the conditions described in the aforementioned license. The license
// is also available online at http://www.enthought.com/licenses/BSD.txt
//
// Thanks for using Enthought open source!
#ifndef KIVA_GL_GRAPHICS_CONTEXT_H
#define KIVA_GL_GRAPHICS_CONTEXT_H

#ifdef _MSC_VER
#pragma warning(disable:4786)
#endif

#ifdef __DARWIN__
    #include <OpenGL/gl.h>
    #include <OpenGL/glu.h>
#else
    #ifdef _MSC_VER
        #include <windows.h>
    #endif
    #include <GL/gl.h>
    #include <GL/glu.h>

    // The following mechanism is necessary in order to use MultiDrawElements
    // on Windows with mingw.
    // 11/24/2010: glMultiDrawElements is not being used right now in the GL
    // backend, so comment this out for the time being, especially since it
    // causes build problems with 64-bit mingw.
    //#ifdef __MINGW32__
    //    #define GL_GLEXT_PROTOTYPES 1
    //    #include <GL/glext.h>
    //    #define MULTI_DRAW_ELEMENTS glMultiDrawElementsEXT
    //#endif
#endif

#include "agg_basics.h"

#include "kiva_gl_compiled_path.h"
#include "kiva_gl_graphics_context_base.h"


namespace kiva_gl
{
    // This function pointer is used by various draw_marker functions
    class gl_graphics_context;
    typedef void(gl_graphics_context::* PathDefinitionFunc)(int);

    class gl_graphics_context : public graphics_context_base
    {
    public:

        gl_graphics_context(int width, int height,
                            kiva_gl::pix_format_e format=kiva_gl::pix_format_rgb24);
        ~gl_graphics_context();

        int width();
        int height();
        int stride();

        //---------------------------------------------------------------
        // GL-specific methods
        //---------------------------------------------------------------
        void gl_init();
        void gl_cleanup();
        void begin_page();
        void gl_render_path(kiva_gl::compiled_path *path, bool polygon=false, bool fill=false);
        void gl_render_points(double** points, bool polygon, bool fill,
                              kiva_gl::draw_mode_e mode = FILL);

        //---------------------------------------------------------------
        // GraphicsContextBase interface
        //---------------------------------------------------------------

        kiva_gl::pix_format_e format();
        void save_state();
        void restore_state();

        //---------------------------------------------------------------
        // Clipping path manipulation
        //---------------------------------------------------------------
        void clip();
        void even_odd_clip();
        void clip_to_rect(double x, double y, double sx, double sy);
        void clip_to_rect(kiva_gl::rect_type &rect);
        void clip_to_rects(double* new_rects, int Nrects);
        void clip_to_rects(kiva_gl::rect_list_type &rects);
        kiva_gl::rect_type transform_clip_rectangle(const kiva_gl::rect_type &rect);
        void clear_clip_path();

        int get_num_clip_regions();
        kiva_gl::rect_type get_clip_region(unsigned int i);

        //---------------------------------------------------------------
        // Painting paths (drawing and filling contours)
        //---------------------------------------------------------------
        void clear(kiva_gl_agg::rgba value=kiva_gl_agg::rgba(1, 1, 1, 1));
        void fill_path();
        void eof_fill_path();
        void stroke_path();
        // empty function; for some reason this is abstract in the base class
        inline void _stroke_path() {}

        void draw_path(draw_mode_e mode=FILL_STROKE);
        void draw_rect(double rect[4], draw_mode_e mode=FILL_STROKE);

        int draw_marker_at_points(double* pts,int Npts,int size,
                                  kiva_gl::marker_e type=kiva_gl::marker_square);

        void draw_path_at_points(double* pts,int Npts,
                                 kiva_gl::compiled_path& marker,
                                 draw_mode_e mode);

        int draw_image(kiva_gl::graphics_context_base* img, double rect[4], bool force_copy=false);
        int draw_image(kiva_gl::graphics_context_base* img);

    protected:

        void draw_display_list_at_pts(GLuint list, double *pts, int Npts,
                                      kiva_gl::draw_mode_e mode,
                                      double x0, double y0);
        void draw_display_list_at_pts(GLuint fill_list, GLuint stroke_list,
                                      double *pts, int Npts,
                                      kiva_gl::draw_mode_e mode,
                                      double x0, double y0);

        // Given a path function, returns two OpenGL display lists representing
        // the list to fill and the list to stroke.  The caller is responsible
        // for calling glDeleteLists on the two.
        // Only the list name of the first list (fill list) will be returned;
        // the stroke list can be accessed by just adding 1.
        GLuint make_marker_lists(kiva_gl::PathDefinitionFunc path_func,
                                 kiva_gl::draw_mode_e mode, int size);

        void circle_path_func(int size);
        void triangle_up_func(int size);
        void triangle_down_func(int size);

        void draw_square(double *pts, int Npts, int size,
                         kiva_gl::draw_mode_e mode, double x0, double y0);
        void draw_diamond(double *pts, int Npts, int size,
                          kiva_gl::draw_mode_e mode, double x0, double y0);
        void draw_crossed_circle(double *pts, int Npts, int size,
                                 kiva_gl::draw_mode_e mode, double x0, double y0);
        void draw_x_marker(double *pts, int Npts, int size,
                           kiva_gl::draw_mode_e mode, double x0, double y0);
        void draw_cross(double *pts, int Npts, int size,
                        kiva_gl::draw_mode_e mode, double x0, double y0);
        void draw_dot(double *pts, int Npts, int size,
                      kiva_gl::draw_mode_e mode, double x0, double y0);
        void draw_pixel(double *pts, int Npts, int size,
                        kiva_gl::draw_mode_e mode, double x0, double y0);

    private:
        int m_width;
        int m_height;
        bool m_gl_initialized;
        kiva_gl::pix_format_e m_pixfmt;
    };
}

#endif  /* KIVA_GL_GRAPHICS_CONTEXT_H */
