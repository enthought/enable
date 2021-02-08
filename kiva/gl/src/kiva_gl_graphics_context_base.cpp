// (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
// All rights reserved.
//
// This software is provided without warranty under the terms of the BSD
// license included in LICENSE.txt and may be redistributed only under
// the conditions described in the aforementioned license. The license
// is also available online at http://www.enthought.com/licenses/BSD.txt
//
// Thanks for using Enthought open source!

#include <assert.h>

#include "agg_path_storage.h"

#include "kiva_gl_exceptions.h"
#include "kiva_gl_graphics_context_base.h"

using namespace kiva_gl;

graphics_context_base::graphics_context_base(interpolation_e interp)
: _image_interpolation(interp)
{
}

graphics_context_base::~graphics_context_base()
{
}

int
graphics_context_base::width()
{
    return 0;
}

int
graphics_context_base::height()
{
    return 0;
}

int
graphics_context_base::stride()
{
    return 1;
}

int
graphics_context_base::bottom_up()
{
    return (this->stride() > 0 ? 0 : 1);
}

kiva_gl::interpolation_e
graphics_context_base::get_image_interpolation()
{
    return this->_image_interpolation;
}

void
graphics_context_base::set_image_interpolation(kiva_gl::interpolation_e interpolation)
{
    this->_image_interpolation = interpolation;
}

//---------------------------------------------------------------
// set graphics_state values
//---------------------------------------------------------------

void
graphics_context_base::set_stroke_color(kiva_gl_agg::rgba& value)
{
    this->state.line_color = value;
}

kiva_gl_agg::rgba&
graphics_context_base::get_stroke_color()
{
    return this->state.line_color;
}

void
graphics_context_base::set_line_width(double value)
{
    this->state.line_width = value;
}

void
graphics_context_base::set_line_join(kiva_gl::line_join_e value)
{
    this->state.line_join = value;
}

void
graphics_context_base::set_line_cap(kiva_gl::line_cap_e value)
{
    this->state.line_cap = value;
}

void
graphics_context_base::set_line_dash(double* pattern, int n, double phase)
{
    this->state.line_dash = kiva_gl::dash_type(phase, pattern, n);
}

void
graphics_context_base::set_blend_mode(kiva_gl::blend_mode_e value)
{
    this->state.blend_mode = value;
}

kiva_gl::blend_mode_e
graphics_context_base::get_blend_mode()
{
    return this->state.blend_mode;
}

void
graphics_context_base::set_fill_color(kiva_gl_agg::rgba& value)
{
    this->state.fill_color = value;
}

kiva_gl_agg::rgba&
graphics_context_base::get_fill_color()
{
    return this->state.fill_color;
}

void
graphics_context_base::set_alpha(double value)
{
    // alpha should be between 0 and 1, so clamp:
    if (value < 0.0)
    {
        value = 0.0;
    }
    else if (value > 1.0)
    {
        value = 1.0;
    }
    this->state.alpha = value;
}

double
graphics_context_base::get_alpha()
{
    return this->state.alpha;
}

void
graphics_context_base::set_antialias(int value)
{
    this->state.should_antialias = value;
}

int
graphics_context_base::get_antialias()
{
    return this->state.should_antialias;
}

void
graphics_context_base::set_miter_limit(double value)
{
    this->state.miter_limit = value;
}

void
graphics_context_base::set_flatness(double value)
{
    this->state.flatness = value;
}

//---------------------------------------------------------------
// save/restore graphics state
//---------------------------------------------------------------

void
graphics_context_base::save_state()
{
    this->state_stack.push(this->state);
    this->path.save_ctm();
}

//---------------------------------------------------------------
// coordinate transform matrix transforms
//---------------------------------------------------------------

void
graphics_context_base::translate_ctm(double x, double y)
{
    this->path.translate_ctm(x, y);
}

void
graphics_context_base::rotate_ctm(double angle)
{
    this->path.rotate_ctm(angle);
}

void
graphics_context_base::scale_ctm(double sx, double sy)
{
    this->path.scale_ctm(sx, sy);
}

void
graphics_context_base::concat_ctm(kiva_gl_agg::trans_affine& m)
{
    this->path.concat_ctm(m);
}

void
graphics_context_base::set_ctm(kiva_gl_agg::trans_affine& m)
{
    this->path.set_ctm(m);
}

kiva_gl_agg::trans_affine
graphics_context_base::get_ctm()
{
    return this->path.get_ctm();
}

//---------------------------------------------------------------
// Sending drawing data to a device
//---------------------------------------------------------------

void
graphics_context_base::flush()
{
    // TODO-PZW: clarify this and other "not sure if anything is needed" functions
    // not sure if anything is needed.
}

void
graphics_context_base::synchronize()
{
    // not sure if anything is needed.
}

//---------------------------------------------------------------
// Page Definitions
//---------------------------------------------------------------

void
graphics_context_base::begin_page()
{
    // not sure if anything is needed.
}

void
graphics_context_base::end_page()
{
    // not sure if anything is needed.
}

//---------------------------------------------------------------
// Path operations
//---------------------------------------------------------------

void
graphics_context_base::begin_path()
{
    this->path.begin_path();
}

void
graphics_context_base::move_to(double x, double y)
{
    this->path.move_to(x, y);
}

void
graphics_context_base::line_to( double x, double y)
{
    this->path.line_to(x, y);
}

void
graphics_context_base::curve_to(double cpx1, double cpy1,
                                double cpx2, double cpy2,
                                double x, double y)
{
    this->path.curve_to(cpx1, cpy1, cpx2, cpy2, x, y);
}

void
graphics_context_base::quad_curve_to(double cpx, double cpy, double x, double y)
{
    this->path.quad_curve_to(cpx, cpy, x, y);
}

void
graphics_context_base::arc(double x, double y, double radius,
                           double start_angle, double end_angle,
                           bool cw)
{
    this->path.arc(x, y, radius, start_angle, end_angle, cw);
}

void
graphics_context_base::arc_to(double x1, double y1, double x2, double y2,
                              double radius)
{
    this->path.arc_to(x1, y1, x2, y2, radius);
}

void
graphics_context_base::close_path()
{
    this->path.close_polygon();
}

void
graphics_context_base::add_path(kiva_gl::compiled_path& other_path)
{
    this->path.add_path(other_path);
}

void
graphics_context_base::lines(double* pts, int Npts)
{
    this->path.lines(pts, Npts);
}

void
graphics_context_base::line_set(double* start, int Nstart, double* end, int Nend)
{
    this->path.line_set(start, Nstart, end, Nend);
}

void
graphics_context_base::rect(double x, double y, double sx, double sy)
{
    this->path.rect(x, y, sx, sy);
}

void
graphics_context_base::rect(kiva_gl::rect_type &rect)
{
    this->path.rect(rect);
}

void
graphics_context_base::rects(double* all_rects, int Nrects)
{
    this->path.rects(all_rects, Nrects);
}

void
graphics_context_base::rects(kiva_gl::rect_list_type &rectlist)
{
    this->path.rects(rectlist);
}

kiva_gl::compiled_path
graphics_context_base::_get_path()
{
    return this->path;
}

kiva_gl::rect_type
graphics_context_base::_get_path_bounds()
{
    double xmin = 0., ymin = 0., xmax = 0., ymax = 0.;
    double x = 0., y = 0.;

    for (unsigned i = 0; i < this->path.total_vertices(); ++i)
    {
        this->path.vertex(i, &x, &y);

        if (i == 0)
        {
            xmin = xmax = x;
            ymin = ymax = y;
            continue;
        }

        if (x < xmin)
        {
            xmin = x;
        }
        else if (xmax < x)
        {
            xmax = x;
        }
        if (y < ymin)
        {
            ymin = y;
        }
        else if (ymax < y)
        {
            ymax = y;
        }
    }

    return kiva_gl::rect_type(xmin, ymin, xmax-xmin, ymax-ymin);
}

kiva_gl_agg::path_storage
graphics_context_base::boundary_path(kiva_gl_agg::trans_affine& affine_mtx)
{
    // Return the path that outlines the image in device space
    // This is used in _draw to specify the device area
    // that should be rendered.
    kiva_gl_agg::path_storage clip_path;
    double p0x = 0;
    double p0y = 0;
    double p1x = this->width();
    double p1y = 0;
    double p2x = this->width();
    double p2y = this->height();
    double p3x = 0;
    double p3y = this->height();

    affine_mtx.transform(&p0x, &p0y);
    affine_mtx.transform(&p1x, &p1y);
    affine_mtx.transform(&p2x, &p2y);
    affine_mtx.transform(&p3x, &p3y);

    clip_path.move_to(p0x, p0y);
    clip_path.line_to(p1x, p1y);
    clip_path.line_to(p2x, p2y);
    clip_path.line_to(p3x, p3y);
    clip_path.close_polygon();
    return clip_path;
}

int
graphics_context_base::draw_image(kiva_gl::graphics_context_base* img)
{
    double tmp[] = {0, 0, img->width(), img->height()};
    return this->draw_image(img, tmp);
}
