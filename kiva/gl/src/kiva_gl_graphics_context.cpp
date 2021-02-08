// (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
// All rights reserved.
//
// This software is provided without warranty under the terms of the BSD
// license included in LICENSE.txt and may be redistributed only under
// the conditions described in the aforementioned license. The license
// is also available online at http://www.enthought.com/licenses/BSD.txt
//
// Thanks for using Enthought open source!

// #ifndef MULTI_DRAW_ELEMENTS
//     #define MULTI_DRAW_ELEMENTS glMultiDrawElements
// #endif

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "kiva_gl_affine_helpers.h"
#include "kiva_gl_exceptions.h"
#include "kiva_gl_rect.h"
#include "kiva_gl_graphics_context.h"

using namespace kiva_gl;

#ifndef CALLBACK
#define CALLBACK
#endif
#ifndef M_PI
#define M_PI 3.1415926535
#endif

#define EXPAND_COLOR(c) c->r, c->g, c->b, (c->a * this->state.alpha)

typedef double VertexType;
struct PointType { VertexType x,y,z; };
typedef std::vector<PointType> PointListType;

static void _submit_path_points(PointListType const & points,
                                bool polygon, bool fill);
static void CALLBACK _combine_callback(GLdouble coords[3], GLdouble *vert_data[4],
                                       GLfloat weight[4], GLdouble **dataOut);
static void CALLBACK _vertex_callback(GLvoid *vertex);

gl_graphics_context::gl_graphics_context(int width, int height,
                                         kiva_gl::pix_format_e format)
: graphics_context_base(kiva_gl::nearest)
, m_width(width)
, m_height(height)
, m_gl_initialized(false)
, m_pixfmt(format)
{
}

gl_graphics_context::~gl_graphics_context()
{
    if (m_gl_initialized)
    {
        this->gl_cleanup();
    }
}

int
gl_graphics_context::width()
{
    return m_width;
}

int
gl_graphics_context::height()
{
    return m_height;
}

int
gl_graphics_context::stride()
{
    return 1;
}

void
gl_graphics_context::gl_init()
{
    glViewport(0, 0, m_width, m_height);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, m_width, 0, m_height, 1, -1);
    glMatrixMode(GL_MODELVIEW);
    //glPushMatrix();
    glLoadIdentity();

    // Use scissors to implement clipping
    glEnable(GL_SCISSOR_TEST);

    // Need to set up blending for antialiasing
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glHint(GL_LINE_SMOOTH_HINT, GL_DONT_CARE);
    glHint(GL_POLYGON_SMOOTH_HINT, GL_DONT_CARE);

    // Clear the clip region
    // This is important.  Since GL maintains a persistent, global context
    // across the application, we may very well inherit the scissor mask
    // from a clip_to_rect() call on a previous GC.
    clip_to_rect(0, 0, m_width, m_height);
}

void
gl_graphics_context::gl_cleanup()
{
    //glMatrixMode(GL_MODELVIEW);
    //glPopMatrix();
}

kiva_gl::pix_format_e
gl_graphics_context::format()
{
    return m_pixfmt;
}

void
gl_graphics_context::save_state()
{
    graphics_context_base::save_state();
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
}

void
gl_graphics_context::restore_state()
{
    if (this->state_stack.size() == 0)
    {
        return;
    }

    this->state = this->state_stack.top();
    this->state_stack.pop();
    this->path.restore_ctm();

    // Restore the clip state:
    // Compute the intersection of all the rects and use those as
    // the clip box
    if (this->state.use_rect_clipping())
    {
        if (this->state.device_space_clip_rects.size() > 0)
        {
            kiva_gl::rect_list_type rects = disjoint_intersect(this->state.device_space_clip_rects);

            // XXX: Right now we don't support disjoint clip rects.  To implement
            // this, we would probably want to use a mask or stencil, or just
            // re-render with each clip rect set as the scissor.
            // XXX: figure out better way to round out the floating-point
            // dimensions for kiva_rect than just casting to int().
            kiva_gl::rect_iterator it = rects.begin();
            glScissor(int(it->x), int(it->y), int(it->w), int(it->h));
        }
    }
    else
    {
        throw clipping_path_unsupported;
    }

    // Restore the transformation matrices
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();

}

void
gl_graphics_context::begin_page()
{
    glClearColor(1.f, 1.f, 1.f, 0.f);
    glClear(GL_COLOR_BUFFER_BIT);
}

void
gl_graphics_context::clip()
{
    throw kiva_gl::not_implemented_error;
}

void
gl_graphics_context::even_odd_clip()
{
    throw kiva_gl::not_implemented_error;
}

void
gl_graphics_context::clip_to_rect(double x, double y, double sx, double sy)
{
    kiva_gl::rect_type tmp(x, y, sx, sy);
    clip_to_rect(tmp);
}

void
gl_graphics_context::clip_to_rect(kiva_gl::rect_type &rect)
{
    this->path.remove_all();
    if (!this->state.use_rect_clipping())
    {
        throw clipping_path_unsupported;
    }

    kiva_gl::rect_type device_rect(transform_clip_rectangle(rect));
    if (this->state.device_space_clip_rects.size() == 1)
    {
        kiva_gl::rect_type old(this->state.device_space_clip_rects.back());
        this->state.device_space_clip_rects.pop_back();
        kiva_gl::rect_type newrect(kiva_gl::disjoint_intersect(old, device_rect));
        if ((newrect.w < 0) || (newrect.h < 0))
        {
            // new clip rectangle doesn't intersect anything, so we push on
            // an empty rect as the new clipping region.
            glScissor(0, 0, 0, 0);
            //printf("NULL intersection area in clip_to_rect\n");
            this->state.device_space_clip_rects.push_back(kiva_gl::rect_type(0, 0, -1, -1));
        }
        else
        {
            glScissor(int(newrect.x), int(newrect.y),
                      int(newrect.w), int(newrect.h));
            this->state.device_space_clip_rects.push_back(newrect);
        }
    }
    else
    {
        // we need to compute the intersection of the new rectangle with
        // the current set of clip rectangles.  we assume that the existing
        // clip_rects are a disjoint set.
        this->state.device_space_clip_rects = kiva_gl::disjoint_intersect(
            this->state.device_space_clip_rects, device_rect);

        if (this->state.device_space_clip_rects.size() == 0)
        {
            glScissor(0, 0, 0, 0);
            //printf("NULL intersection area in clip_to_rect\n");
            this->state.device_space_clip_rects.push_back(kiva_gl::rect_type(0, 0, -1, -1));
        }
        else
        {
            kiva_gl::rect_list_type rects = disjoint_intersect(this->state.device_space_clip_rects);

            // XXX: Right now we don't support disjoint clip rects.
            // (same problem as in restore_state())
            kiva_gl::rect_iterator it = rects.begin();
            glScissor(int(it->x), int(it->y), int(it->w), int(it->h));
            if (rects.size() > 1)
            {
                //printf("Warning: more than 1 clip rect in clip_to_rect()\n");
            }
        }
    }
}

void
gl_graphics_context::clip_to_rects(double* new_rects, int Nrects)
{
    printf("Clip to rects() unsupported\n");
}

void
gl_graphics_context::clip_to_rects(kiva_gl::rect_list_type &rects)
{
    printf("Clip to rects() unsupported\n");
}

void
gl_graphics_context::clear_clip_path()
{
    // clear the existing clipping paths
    this->state.clipping_path.remove_all();
    this->state.device_space_clip_rects.clear();

    // set everything visible again.
    glScissor(0, 0, m_width, m_height);

    // store the new clipping rectangle back into the first
    // rectangle of the graphics state clipping rects.
    this->state.device_space_clip_rects.push_back(kiva_gl::rect_type(0, 0, m_width, m_height));
}

// XXX: This is cut and paste from graphics_context.h; refactor into base
// class.
kiva_gl::rect_type
gl_graphics_context::transform_clip_rectangle(const kiva_gl::rect_type &rect)
{
    // This only works if the ctm doesn't have any rotation.
    // otherwise, we need to use a clipping path. Test for this.
    kiva_gl_agg::trans_affine tmp(this->path.get_ctm());
    if (!only_scale_and_translation(tmp))
    {
        throw kiva_gl::ctm_rotation_error;
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

    return kiva_gl::rect_type(x, y, x2-x, y2-y);
}

int
gl_graphics_context::get_num_clip_regions()
{
    return this->state.device_space_clip_rects.size();
}

kiva_gl::rect_type
gl_graphics_context::get_clip_region(unsigned int i)
{
    throw kiva_gl::not_implemented_error;
}

void
gl_graphics_context::clear(kiva_gl_agg::rgba value)
{
    glClearColor(float(value.r), float(value.g), float(value.b), float(value.a));
    glClear(GL_COLOR_BUFFER_BIT);
}

void
gl_graphics_context::fill_path()
{
    draw_path(FILL);
}

void
gl_graphics_context::eof_fill_path()
{
    draw_path(EOF_FILL);
}

void
gl_graphics_context::stroke_path()
{
    draw_path(STROKE);
}

void
gl_graphics_context::gl_render_path(kiva_gl::compiled_path *path, bool polygon, bool fill)
{
    if ((path == NULL) || (path->total_vertices() == 0))
    {
        return;
    }

    unsigned command = 0;
    PointListType pointList;

    // Set the matrix mode so we support move_to commands
    glMatrixMode(GL_MODELVIEW);

    // Records the last move_to command position so that when
    // we finally encounter the first line_to, we can use this
    // vertex as the starting vertex.
    bool first_vertex_drawn = false;
    PointType v0 = {0.f, 0.f, 0.f};
    PointType v = {0.f, 0.f, 0.f};
    PointType vv = {0.f, 0.f, 0.f};
    VertexType c1x, c1y, ccx, ccy, c2x, c2y, c3x, c3y;
    VertexType t, t2, t3, u, u2, u3;
    unsigned int j;
    unsigned int _Npoints = 100;

    // make space for points
    pointList.reserve(path->total_vertices());

    for (unsigned int i=0; i < path->total_vertices(); ++i)
    {
        command = path->vertex(i, &v.x, &v.y);
        switch (command & kiva_gl_agg::path_cmd_mask)
        {
        case kiva_gl_agg::path_cmd_line_to:
            if (!first_vertex_drawn)
            {
                pointList.push_back(v0);
                first_vertex_drawn = true;
            }
            pointList.push_back(v);
            break;

        case kiva_gl_agg::path_cmd_end_poly:
            // We shouldn't need to do anything because if this is a closed path
            //
            //if (command & kiva_gl_agg::path_flags_close)
            //    glVertex2f(x0, y0);
            break;

        case kiva_gl_agg::path_cmd_curve3:
            // FIXME: refactor!
            if (!first_vertex_drawn)
            {
                pointList.push_back(v0);
                first_vertex_drawn = true;
            }
            path->vertex(i+1, &ccx, &ccy);
            path->vertex(i+2, &c3x, &c3y);
            i += 2;
            c1x = (v.x + ccx + ccx) / 3.0;
            c1y = (v.y + ccy + ccy) / 3.0;
            c2x = (c3x + ccx + ccx) / 3.0;
            c2y = (c3y + ccy + ccy) / 3.0;
            for (j=1; j<=_Npoints; ++j)
            {
                t = ((VertexType)j) / _Npoints;
                t2 = t*t;
                t3 = t2*t;
                u = 1 - t;
                u2 = u*u;
                u3 = u2*u;
                vv.x = v.x * u3 + 3*(c1x*t*u2 + c2x*t2*u) + c3x*t3;
                vv.y = v.y * u3 + 3*(c1y*t*u2 + c2y*t2*u) + c3y*t3;
                pointList.push_back(vv);
            }
            break;

        case kiva_gl_agg::path_cmd_curve4:
            if (!first_vertex_drawn)
            {
                pointList.push_back(v0);
                first_vertex_drawn = true;
            }
            // The current point is implicitly the first control point
            v0 = pointList.back();
            c1x = v.x; c1y = v.y;
            v.x = v0.x; v.y = v0.y;
            path->vertex(i+1, &c2x, &c2y);
            path->vertex(i+2, &c3x, &c3y);
            i += 2;
            for (j=1; j<=_Npoints; ++j)
            {
                t = ((VertexType)j) / _Npoints;
                t2 = t*t;
                t3 = t2*t;
                u = 1 - t;
                u2 = u*u;
                u3 = u2*u;
                vv.x = v.x * u3 + 3*(c1x*t*u2 + c2x*t2*u) + c3x*t3;
                vv.y = v.y * u3 + 3*(c1y*t*u2 + c2y*t2*u) + c3y*t3;
                pointList.push_back(vv);
            }
            break;

        // The following commands are ignored.
        case kiva_gl_agg::path_cmd_move_to:
            if (!pointList.empty())
            {
                // do a full glBegin/glEnd sequence for the points in the buffer
                _submit_path_points(pointList, polygon, fill);
                // flush
                pointList.clear();
            }
            v0.x = v.x;
            v0.y = v.y;
            first_vertex_drawn = false;
            break;

        case kiva_gl_agg::path_cmd_ubspline:
            break;

        // XXX: This case number is already used??
        //case kiva_gl_agg::path_cmd_mask:
        //    break;

        // Unsupported
        // XXX: We need to have better error handling/reporting from the C++
        // layer up to the Python layer.
        case kiva_gl_agg::path_cmd_catrom:
        case kiva_gl_agg::path_cmd_curveN:
            break;

        }
    }

    // submit the points
    if (!pointList.empty())
    {
        _submit_path_points(pointList, polygon, fill);
    }
}

void
gl_graphics_context::gl_render_points(double** points, bool polygon,
                                      bool fill, kiva_gl::draw_mode_e mode)
{
}

void
gl_graphics_context::draw_path(draw_mode_e mode)
{
    // XXX: This is a direct transcription from basecore2d.  The algorithm
    // and approach can probably be improved tremendously for OpenGL.

    kiva_gl_agg::rgba *line_color = &this->state.line_color;
    kiva_gl_agg::rgba *fill_color = &this->state.fill_color;

    // CNP
    if (this->state.should_antialias)
    {
        glEnable(GL_LINE_SMOOTH);
        glEnable(GL_POLYGON_SMOOTH);
    }
    else
    {
        glDisable(GL_LINE_SMOOTH);
        glDisable(GL_POLYGON_SMOOTH);
    }

    // Check to see if we have closed polygons
    typedef kiva_gl_agg::path_storage::container_type::value_type VertexType;
    unsigned numvertices = this->path.total_vertices();
    bool polygon = false;
    if (numvertices > 1)
    {
        // Get the first vertex
        VertexType x0, y0, xf, yf;
        this->path.vertex(0, &x0, &y0);

        // Go backwards from the last vertex until we find an actual line_to
        // or curve3 or curve4 comand.
        for (int i=numvertices-1; i>0; --i)
        {
            unsigned cmd = this->path.vertex(i, &xf, &yf);
            if (((cmd & kiva_gl_agg::path_cmd_mask) == kiva_gl_agg::path_cmd_curve3) ||
                ((cmd & kiva_gl_agg::path_cmd_mask) == kiva_gl_agg::path_cmd_curve4) ||
                ((cmd & kiva_gl_agg::path_cmd_mask) == kiva_gl_agg::path_cmd_line_to))
            {
                if ((x0 == xf) && (y0 == yf))
                {
                    polygon = true;
                }
                break;
            }

            if ((cmd & kiva_gl_agg::path_cmd_mask) == kiva_gl_agg::path_cmd_end_poly)
            {
                polygon = true;
                break;
            }
        }
    }

    // Fill the path, if necessary
    if (mode != STROKE)
    {
        // device_update_fill_state
        glColor4f(EXPAND_COLOR(fill_color));

        // call gl_render_path()
        gl_render_path(&this->path, true, true);
    }

    // Stroke the path, if necessary
    if (mode != FILL)
    {
        // CNP
        // device_update_line_state
        glColor4f(EXPAND_COLOR(line_color));
        glLineWidth(this->state.line_width);

        if (this->state.line_dash.is_solid())
        {
            glDisable(GL_LINE_STIPPLE);
        }
        else
        {
            glDisable(GL_LINE_STIPPLE);
        }

        gl_render_path(&this->path, polygon, false);
    }

    this->path.remove_all();
}

void
gl_graphics_context::draw_rect(double rect[4], draw_mode_e mode)
{
    kiva_gl_agg::rgba *line_color = &this->state.line_color;
    kiva_gl_agg::rgba *fill_color = &this->state.fill_color;

    // CNP
    if (this->state.should_antialias)
    {
        glEnable(GL_LINE_SMOOTH);
        glEnable(GL_POLYGON_SMOOTH);
    }
    else
    {
        glDisable(GL_LINE_SMOOTH);
        glDisable(GL_POLYGON_SMOOTH);
    }

    this->path.get_ctm().translation(rect, rect+1);

    // Fill the rect first
    if (mode != STROKE)
    {
        glColor4f(EXPAND_COLOR(fill_color));
        glRectf(rect[0], rect[1], rect[0]+rect[2], rect[1]+rect[3]);
    }

    // Stroke the path
    if (mode != FILL)
    {
        // CNP
        glColor4f(EXPAND_COLOR(line_color));
        glLineWidth(this->state.line_width);

        if (this->state.line_dash.is_solid())
        {
            glDisable(GL_LINE_STIPPLE);
        }
        else
        {
            glDisable(GL_LINE_STIPPLE);
        }

        glBegin(GL_LINE_LOOP);
        glVertex2f(rect[0], rect[1]);
        glVertex2f(rect[0], rect[1] + rect[3]);
        glVertex2f(rect[0] + rect[2], rect[1] + rect[3]);
        glVertex2f(rect[0] + rect[2], rect[1]);
        glEnd();
    }
    this->path.remove_all();
}

int
gl_graphics_context::draw_marker_at_points(double *pts, int Npts,
                                           int size, kiva_gl::marker_e type)
{
    kiva_gl_agg::rgba *line_color = &this->state.line_color;
    kiva_gl_agg::rgba *fill_color = &this->state.fill_color;
    bool do_fill = (fill_color->a != 0);
    bool do_stroke = ((line_color->a != 0) && (this->state.line_width > 0.0));

    if (do_stroke)
    {
        glLineWidth(this->state.line_width);
    }

    // Get the current origin
    double x0=0.0, y0=0.0;
    this->path.get_ctm().translation(&x0, &y0);

    kiva_gl::draw_mode_e draw_mode = FILL;
    if (do_fill & !do_stroke)
    {
        draw_mode = FILL;
    }
    else if (do_stroke & !do_fill)
    {
        draw_mode = STROKE;
    }
    else if (do_fill & do_stroke)
    {
        draw_mode = FILL_STROKE;
    }
    GLuint fill_list, stroke_list;
    bool list_created = false;

    switch (type)
    {
    // Simple paths that only need to be stroked
    case kiva_gl::marker_x:
        draw_x_marker(pts, Npts, size, draw_mode, x0, y0);
        break;

    case kiva_gl::marker_cross:
        draw_cross(pts, Npts, size, draw_mode, x0, y0);
        break;

    case kiva_gl::marker_dot:
        draw_dot(pts, Npts, size, draw_mode, x0, y0);
        break;

    case kiva_gl::marker_pixel:
        draw_pixel(pts, Npts, size, draw_mode, x0, y0);
        break;

    // Paths that need to be filled and stroked
    // There are experimental approaches taken for drawing squares and
    // diamonds, so they are in their own block here.  There's no reason
    // why they cannot be treated in the same way as the circle and
    // triangle markers.
    case kiva_gl::marker_square:
        draw_square(pts, Npts, size, draw_mode, x0, y0);
        break;

    case kiva_gl::marker_diamond:
        draw_diamond(pts, Npts, size, draw_mode, x0, y0);
        break;

    case kiva_gl::marker_crossed_circle:
        draw_crossed_circle(pts, Npts, size, draw_mode, x0, y0);
        break;

    case kiva_gl::marker_circle:
        fill_list = make_marker_lists(&kiva_gl::gl_graphics_context::circle_path_func, draw_mode, size);
        list_created = true;
        // Fall through to next case
    case kiva_gl::marker_triangle_up:
        if (!list_created)
        {
            fill_list = make_marker_lists(&kiva_gl::gl_graphics_context::triangle_up_func, draw_mode, size);
            list_created = true;
        }
        // Fall through to next case
    case kiva_gl::marker_triangle_down:
        if (!list_created)
        {
            fill_list = make_marker_lists(&kiva_gl::gl_graphics_context::triangle_down_func, draw_mode, size);
            list_created = true;
        }
        stroke_list = fill_list + 1;
        draw_display_list_at_pts(fill_list, stroke_list, pts, Npts, draw_mode, x0, y0);
        glDeleteLists(fill_list, 2);
        break;

    default:
        return 0;
    }
    // Indicate success
    return 1;
}

void
gl_graphics_context::draw_path_at_points(double *pts, int Npts,
                                         kiva_gl::compiled_path &marker,
                                         draw_mode_e mode)
{
    return;
}

int
gl_graphics_context::draw_image(kiva_gl::graphics_context_base* img,
                                double rect[4], bool force_copy)
{
    return 0;
}

int gl_graphics_context::draw_image(kiva_gl::graphics_context_base* img)
{
    return 0;
}

//---------------------------------------------------------------------------
// Marker drawing methods
//---------------------------------------------------------------------------

void
gl_graphics_context::draw_display_list_at_pts(GLuint list, double *pts, int Npts,
                                              kiva_gl::draw_mode_e mode,
                                              double x0, double y0)
{
    draw_display_list_at_pts(list, list, pts, Npts, mode, x0, y0);
}

void
gl_graphics_context::draw_display_list_at_pts(GLuint fill_list, GLuint stroke_list,
                                              double *pts, int Npts,
                                              kiva_gl::draw_mode_e mode,
                                              double x0, double y0)
{
    kiva_gl_agg::rgba *colors[2] = { &this->state.fill_color, &this->state.line_color };
    GLuint lists[2] = { fill_list, stroke_list };
    float x = 0.f, y = 0.f;
    for (int pass=0; pass < 2; ++pass)
    {
        if (((pass == 0) && ((mode == FILL) || (mode == FILL_STROKE))) ||
            ((pass == 1) && ((mode == STROKE) || (mode == FILL_STROKE))))
        {
            glColor4f(EXPAND_COLOR(colors[pass]));
            for (int i=0; i < Npts; ++i)
            {
                x = pts[i*2] + x0;
                y = pts[i*2 + 1] + y0;
                glTranslatef(x, y, 0.0);
                glCallList(lists[pass]);
                glTranslatef(-x, -y, 0.0);
            }
        }
    }

#if 0
    if ((mode == FILL) || (mode == FILL_STROKE))
    {
        glColor4f(EXPAND_COLOR(fill_color));
        for (int i=0; i < Npts; ++i)
        {
            x = pts[i*2] + x0;
            y = pts[i*2 + 1] + y0;
            glTranslatef(x, y, 0.0);
            glCallList(stroke_list);
            glTranslatef(-x, -y, 0.0);
        }
    }
    if ((mode == STROKE) || (mode == FILL_STROKE))
    {
        glColor4f(EXPAND_COLOR(line_color));
        for (int i=0; i < Npts; ++i)
        {
            x = pts[i*2] + x0;
            y = pts[i*2 + 1] + y0;
            glTranslatef(x, y, 0.0);
            glCallList(fill_list);
            glTranslatef(-x, -y, 0.0);
        }
    }
#endif
}

GLuint
gl_graphics_context::make_marker_lists(PathDefinitionFunc path_func,
                                       kiva_gl::draw_mode_e mode,
                                       int size)
{
    GLuint fill_list = glGenLists(2);
    GLuint stroke_list = fill_list + 1;
    for (int dummy=0; dummy < 2; ++dummy)
    {
        if (dummy == 0)
        {
            glNewList(fill_list, GL_COMPILE);
            glBegin(GL_POLYGON);
        }
        else
        {
            glNewList(stroke_list, GL_COMPILE);
            glBegin(GL_LINE_LOOP);
        }
        ((this)->*(path_func))(size);
        glEnd();
        glEndList();
    }

    return fill_list;
}

void
gl_graphics_context::draw_square(double *pts, int Npts, int size,
                                 kiva_gl::draw_mode_e mode,
                                 double x0, double y0)
{
    kiva_gl_agg::rgba *line_color = &this->state.line_color;
    kiva_gl_agg::rgba *fill_color = &this->state.fill_color;

    // We build up a VertexArray of the vertices of all the squares.
    // We then use glDrawElements with GL_QUADS or GL_LINE_LOOP to fill
    // and stroke the markers.

    // The vertex array contains all the vertices in all the rects.
    // The convention is that each rect's vertices are stored
    // clockwise, starting with the lower-left vertex.
    GLdouble *vertices = new GLdouble[Npts*4*2];

    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(2, GL_DOUBLE, 0, vertices);
    for (int i=0; i<Npts; ++i)
    {
        int rect = i * 4 * 2;
        double x = pts[i*2] - size/2.0 + x0;
        double y = pts[i*2+1] - size/2.0 + y0;
        // Set the x-coords of the left vertices
        vertices[rect] = vertices[rect+2] = x;
        // Set the x-coords of the right vertices
        vertices[rect+4] = vertices[rect+6] = x + size;
        // Set the y-coords of the bottom vertices
        vertices[rect+1] = vertices[rect+7] = y;
        // Set the y-coords of the top vertices
        vertices[rect+3] = vertices[rect+5] = y + size;
    }

    if ((mode == FILL) || (mode == FILL_STROKE))
    {
        glColor4f(EXPAND_COLOR(fill_color));
        GLuint *indices = new GLuint[Npts*4];
        for (int i=0; i<Npts*4; ++i)
        {
            indices[i] = i;
        }
        glDrawElements(GL_QUADS, Npts*4, GL_UNSIGNED_INT, indices);
        delete[] indices;
    }
    if ((mode == STROKE) || (mode == FILL_STROKE))
    {
        glColor4f(EXPAND_COLOR(line_color));

// To use glMultiDrawElements in a robust manner, we'll need to do a bunch
// of extension function pointer juggling.  Avoid this for now and just
// use the more brute-force approach.
#if 0
        // To use glMultiDrawElements, we need to have a GLvoid** of
        // indices.  To avoid a lot of unnecessary memory allocation, we
        // just allocate a single array of all the indices and set up
        // the top-level array of arrays to point into it.
        GLvoid **indices = new GLvoid*[Npts];
        GLuint *realindices = new GLuint[Npts*4];
        GLsizei *counts = new GLsizei[Npts];
        for (int i=0; i<Npts; ++i)
        {
            realindices[i*4] = i*4;
            realindices[i*4+1] = i*4 + 1;
            realindices[i*4+2] = i*4 + 2;
            realindices[i*4+3] = i*4 + 3;
            indices[i] = (GLvoid*)(realindices + i*4);
            counts[i] = 4;
        }
        MULTI_DRAW_ELEMENTS(GL_LINE_LOOP, counts, GL_UNSIGNED_INT,
                            (const GLvoid**)indices, Npts);

        delete[] counts;
        delete[] realindices;
        delete[] indices;
#elif 1
        GLuint indices[4] = {0, 1, 2, 3};
        // We can theoretically use glMultiDrawElements, but that requires
        // us to build up a useless array of counts, as well as a 2D array
        // of indices.  This is more straightforward.

        for (int i=0; i<Npts; ++i)
        {
            glDrawElements(GL_LINE_LOOP, 4, GL_UNSIGNED_INT, indices);
            indices[0] += 4;
            indices[1] += 4;
            indices[2] += 4;
            indices[3] += 4;
        }
#endif
    }

    glDisableClientState(GL_VERTEX_ARRAY);
    delete[] vertices;
}

void
gl_graphics_context::draw_diamond(double *pts, int Npts, int size,
                                  kiva_gl::draw_mode_e mode,
                                  double x0, double y0)
{
    kiva_gl_agg::rgba *line_color = &this->state.line_color;
    kiva_gl_agg::rgba *fill_color = &this->state.fill_color;

    // Each marker consists of four vertices in this order: left, top, right, bottom.
    GLdouble *vertices = new GLdouble[Npts * 4 * 2];
    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(2, GL_DOUBLE, 0, vertices);

    float s = size / 2.0;
    for (int i=0; i<Npts; ++i)
    {
        int ndx = i * 4 * 2;
        double x = pts[2*i] + x0;
        double y = pts[2*i + 1] + y0;

        // Set the x-coords of the top and bottom vertices
        vertices[ndx+2] = vertices[ndx+6] = x;
        // Set the y-coords of the left and right vertices
        vertices[ndx+1] = vertices[ndx+5] = y;

        // x-coords of left and right vertices
        vertices[ndx+0] = x-s;
        vertices[ndx+4] = x+s;

        // y-coords of top and bottom vertices
        vertices[ndx+3] = y+s;
        vertices[ndx+7] = y-s;
    }
    if ((mode == FILL) || (mode == FILL_STROKE))
    {
        glColor4f(EXPAND_COLOR(fill_color));
        GLuint *indices = new GLuint[Npts*4];
        for (int i=0; i<Npts*4; ++i)
        {
            indices[i] = i;
        }
        glDrawElements(GL_QUADS, Npts*4, GL_UNSIGNED_INT, indices);
        delete[] indices;
    }
    if ((mode == STROKE) || (mode == FILL_STROKE))
    {
        glColor4f(EXPAND_COLOR(line_color));
        GLuint indices[4] = {0, 1, 2, 3};
        // We can theoretically use glMultiDrawElements, but that requires
        // us to build up a useless array of counts, as well as a 2D array
        // of indices.  This is more straightforward.

        for (int i=0; i<Npts; ++i)
        {
            glDrawElements(GL_LINE_LOOP, 4, GL_UNSIGNED_INT, indices);
            indices[0] += 4;
            indices[1] += 4;
            indices[2] += 4;
            indices[3] += 4;
        }
    }

    glDisableClientState(GL_VERTEX_ARRAY);
    delete[] vertices;
}

void
gl_graphics_context::circle_path_func(int size)
{
    // Compute the points on the circle; note that size is diameter and not radius
    int numCirclePts = int(M_PI * size);
    double theta = 0.0;

    for (int i=0; i < numCirclePts; ++i)
    {
        theta = 2.0 * M_PI / numCirclePts * i;
        glVertex2f(size / 2.0 * cos(theta), size / 2.0 * sin(theta));
    }
}

void
gl_graphics_context::triangle_up_func(int size)
{
    float h = size / (sqrt(3.f) / 2.0);
    glVertex2f(-size/2.0, -h/3.0);
    glVertex2f(0, h * 2.0 / 3.0);
    glVertex2f(size/2.0, -h/3.0);
}

void
gl_graphics_context::triangle_down_func(int size)
{
    float h = size / (sqrt(3.f) / 2.0);
    glVertex2f(-size/2.0, h/3.0);
    glVertex2f(size/2.0, h/3.0);
    glVertex2f(0, -h * 2.0 / 3.0);
}

void
gl_graphics_context::draw_crossed_circle(double *pts, int Npts, int size,
                                         kiva_gl::draw_mode_e mode,
                                         double x0, double y0)
{
    // Draw the circle
    GLuint fill_list = make_marker_lists(&kiva_gl::gl_graphics_context::circle_path_func,
                                         mode, size);
    GLuint stroke_list = fill_list + 1;
    draw_display_list_at_pts(fill_list, stroke_list, pts, Npts, mode, x0, y0);
    glDeleteLists(fill_list, 2);

    // Draw the "X"
    draw_x_marker(pts, Npts, size, STROKE, x0, y0);
}

void
gl_graphics_context::draw_x_marker(double *pts, int Npts, int size,
                                   kiva_gl::draw_mode_e mode,
                                   double x0, double y0)
{
    if (mode == FILL)
    {
        return;
    }

    float s = size / 2.0;
    GLuint marker_list = glGenLists(1);

    // Create the marker
    glNewList(marker_list, GL_COMPILE);
    glBegin(GL_LINES);
    glVertex2f(-s, -s);
    glVertex2f(s, s);
    glVertex2f(-s, s);
    glVertex2f(s, -s);
    glEnd();
    glEndList();

    draw_display_list_at_pts(marker_list, pts, Npts, mode, x0, y0);
    glDeleteLists(marker_list, 1);
}

void
gl_graphics_context::draw_cross(double *pts, int Npts, int size,
                                kiva_gl::draw_mode_e mode,
                                double x0, double y0)
{
    if (mode == FILL)
    {
        return;
    }

    float s = size / 2.0;
    GLuint marker_list = glGenLists(1);

    // Create the marker
    glNewList(marker_list, GL_COMPILE);
    glBegin(GL_LINES);
    glVertex2f(-s, 0);
    glVertex2f(s, 0);
    glVertex2f(0, -s);
    glVertex2f(0, s);
    glEnd();
    glEndList();

    draw_display_list_at_pts(marker_list, pts, Npts, mode, x0, y0);
    glDeleteLists(marker_list, 1);
}

void
gl_graphics_context::draw_dot(double *pts, int Npts, int size,
                              kiva_gl::draw_mode_e mode,
                              double x0, double y0)
{
}

void
gl_graphics_context::draw_pixel(double *pts, int Npts, int size,
                                kiva_gl::draw_mode_e mode,
                                double x0, double y0)
{
    kiva_gl_agg::rgba *line_color = &this->state.line_color;
    glColor4f(EXPAND_COLOR(line_color));

    glBegin(GL_POINTS);
    for (int i=0; i < Npts; ++i)
    {
        glVertex2f(pts[i*2] + x0, pts[i*2+1] + y0);
    }
    glEnd();
}

void
_submit_path_points(PointListType const & points, bool polygon, bool fill)
{
    // Uncomment this when we turn the glPolygonMode calls back on (below)
    //glPushAttrib(GL_POLYGON_BIT);
    if (polygon)
    {
        if (fill)
        {
#if defined(_MSC_VER) || defined(__MINGW32__)
            typedef void (__stdcall*cbFunc)(void);
#else
            typedef void (*cbFunc)();
#endif
            GLUtesselator* pTess = gluNewTess();
            gluTessCallback(pTess, GLU_TESS_VERTEX, (cbFunc)&_vertex_callback);
            gluTessCallback(pTess, GLU_TESS_BEGIN, (cbFunc)&glBegin);
            gluTessCallback(pTess, GLU_TESS_END, (cbFunc)&glEnd);
            gluTessCallback(pTess, GLU_TESS_COMBINE, (cbFunc)&_combine_callback);
            gluTessBeginPolygon(pTess, NULL);
            gluTessBeginContour(pTess);

            // XXX: For some reason setting the polygon mode breaks pyglet's
            // font rendering.  It doesn't really have an effect on any of
            // Kiva's rendering right now, so it's commented out for now.
            //glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

            for (int i=0; i < points.size(); ++i)
            {
                VertexType * pV = (VertexType *)&points[i];
                gluTessVertex(pTess, (GLdouble*)pV, (GLvoid*)pV);
            }

            gluTessEndContour(pTess);
            gluTessEndPolygon(pTess);
            gluDeleteTess(pTess);
        }
        else
        {
            glBegin(GL_LINE_LOOP);
            //glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

            for (int i=0; i < points.size(); ++i)
            {
                glVertex2dv((VertexType *)&points[i]);
            }

            glEnd();
        }
    }
    else
    {
        glBegin(GL_LINE_STRIP);

        for (int i=0; i < points.size(); ++i)
        {
            glVertex2dv((VertexType *)&points[i]);
        }

        glEnd();
    }

    //glPopAttrib();
}

void
CALLBACK
_combine_callback(GLdouble coords[3], GLdouble *vert_data[4],
                  GLfloat weight[4], GLdouble **dataOut)
{
    GLdouble *vertex = (GLdouble *)malloc(3 * sizeof(GLdouble));
    vertex[0] = coords[0];
    vertex[1] = coords[1];
    vertex[2] = coords[2];

    *dataOut = vertex;
}

void
CALLBACK
_vertex_callback(GLvoid *vertex)
{
    GLdouble *ptr = (GLdouble *)vertex;
    glVertex3dv(ptr);
}
