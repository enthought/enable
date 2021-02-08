// (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
// All rights reserved.
//
// This software is provided without warranty under the terms of the BSD
// license included in LICENSE.txt and may be redistributed only under
// the conditions described in the aforementioned license. The license
// is also available online at http://www.enthought.com/licenses/BSD.txt
//
// Thanks for using Enthought open source!
#include "agg_bezier_arc.h"
#include "kiva_compiled_path.h"
#include "kiva_basics.h"
#include <math.h>
#include <assert.h>

using namespace kiva;

void compiled_path::remove_all()
{
    //agg24::path_storage::remove_all();
    // fix me: call to base:: to appease VC++6.0
    this->base::remove_all();
    this->_has_curves = false;
    //ptm = agg24::trans_affine();
}

void compiled_path::begin_path()
{
    this->remove_all();
}

void compiled_path::close_path()
{
    this->close_polygon();
}

void compiled_path::move_to(double x, double y)
{
    this->ptm.transform(&x, &y);
    // fix me: call to base:: to appease VC++6.0
    this->base::move_to(x,y);
}

void compiled_path::line_to(double x, double y)
{
    this->ptm.transform(&x, &y);
    // fix me: call to base:: to appease VC++6.0
    this->base::line_to(x,y);
}

void compiled_path::quad_curve_to(double x_ctrl, double y_ctrl,
                                  double x_to, double y_to)
{
    this->ptm.transform(&x_ctrl, &y_ctrl);
    this->ptm.transform(&x_to, &y_to);
    // fix me: call to base:: to appease VC++6.0
    this->base::curve3(x_ctrl,y_ctrl,x_to,y_to);
    this->_has_curves = true;
}

void compiled_path::curve_to(double x_ctrl1, double y_ctrl1,
                             double x_ctrl2, double y_ctrl2,
                             double x_to, double y_to)
{
    this->ptm.transform(&x_ctrl1, &y_ctrl1);
    this->ptm.transform(&x_ctrl2, &y_ctrl2);
    this->ptm.transform(&x_to, &y_to);
    // fix me: call to base:: to appease VC++6.0
    this->base::curve4(x_ctrl1,y_ctrl1,
                       x_ctrl2,y_ctrl2,
                       x_to,y_to);
    this->_has_curves = true;
}

void compiled_path::arc(double x, double y, double radius, double start_angle,
                        double end_angle, bool cw)
{
    // Rather than try to transform the center and scale the axes correctly,
    // we'll just create an untransformed agg curve, grab its Bezier control
    // points, transform them, and manually add them to the path.
    double sweep_angle = end_angle - start_angle;
    if (cw)
    {
        sweep_angle = -(2*agg24::pi - sweep_angle);
    }
    agg24::bezier_arc aggarc(x, y, radius, radius, start_angle, sweep_angle);

    // Now manually transform each vertex and add it.  For some reason, trying
    // to transform aggarc in place and then using this->base::add_path()
    // causes an access violation if cw=true (but works fine if cw=false).
    int numverts = aggarc.num_vertices();
    container_type& vertices = this->vertices();
    double vx, vy;
    unsigned int cmd;
    aggarc.rewind(0);
    for (int i = 0; i <= numverts/2; i++)
    {
        cmd = aggarc.vertex(&vx, &vy);
        if (!agg24::is_stop(cmd))
        {
            this->ptm.transform(&vx, &vy);
            vertices.add_vertex(vx, vy, cmd);
        }
    }

    this->_has_curves = true;
}

void compiled_path::arc_to(double x1, double y1, double x2, double y2,
                           double radius)
{
    // We have to do some work above and beyond what Agg offers.  The Agg
    // arc_to() happily creates rotated elliptical arcs, but to match the
    // DisplayPDF spec, we need to compute the correct tangent points on
    // the tangent lines defined by (cur_x,cur_y), (x1,y1), and (x2,y2) such
    // that a circular arc of the given radius will be created.

    // The general approach is to transform the coordinates of the three
    // points so that x1,y1 is at the origin, x0,y0 is to the right of x1,y1,
    // and y0==y1.  This should be just a translation followed by a rotation.
    // We then compute the relative position of the circle's center as well
    // as the start angle and then inverse transform these back.  (The angular
    // sweep of the arc is unchanged.)

    double x0=0, y0=0;
    this->last_vertex(&x0, &y0);
    this->ptm.inverse_transform(&x0, &y0);

    // Calculate the offset and rotation so that x1,y1, is at the origin (0,0),
    // and x0, y0 sites on the positive x axis (right side of x1,y1).
    agg24::trans_affine_translation xform(-x1, -y1);
    double xform_angle = -atan2(y0-y1, x0-x1);
    if (!kiva::almost_equal(fmod(xform_angle, 2*agg24::pi), 0.0))
    {
        xform *= agg24::trans_affine_rotation(xform_angle);
    }

    // Transform and rotate the points.
    xform.transform(&x0, &y0);
    xform.transform(&x1, &y1);
    xform.transform(&x2, &y2);

    assert(kiva::almost_equal(y1, 0.0));
    assert(kiva::almost_equal(x1, 0.0));

    double cx, cy;  // location of circle's center
    double center_angle = atan2(y2, x2) / 2;
    bool sweep_flag = (center_angle >= 0) ? false : true;
    double hypotenuse = fabs(radius / sin(center_angle));
    cx = hypotenuse * cos(center_angle);
    cy = hypotenuse * sin(center_angle);

    // determine if we need to draw a line to the first tangent point
    // from the current pen position.
    if (!kiva::almost_equal(x0, cx))
    {
        x0 = cx;
        xform.inverse_transform(&x0, &y0);
        this->line_to(x0, y0);
    }
    else
    {
        xform.inverse_transform(&x0, &y0);
    }

    // determine the second tangent point
    double point2_scale = cx / sqrt(x2*x2 + y2*y2);
    x2 *= point2_scale;
    y2 *= point2_scale;
    xform.inverse_transform(&x2, &y2);
    agg24::bezier_arc_svg aggarc(x0, y0, radius, radius, 0.0, false, sweep_flag, x2, y2);

    int numverts = aggarc.num_vertices();
    double *vertices = aggarc.vertices();
    double *v = NULL;
    for (int i = 0; i <= numverts/2; i++)
    {
        v = vertices + i*2;
        this->ptm.transform(v, v+1);
    }

    // I believe join_path is equivalent to the old add_path() with solid_path=true
    this->join_path(aggarc, 0);
    // This is the alternative call.
    //this->concat_path(aggarc, 0);

    this->_has_curves = true;
}


void compiled_path::add_path(compiled_path& other_path)
{
    container_type& vertices = this->vertices();
    double x=0.0;
    double y=0.0;
    unsigned cmd;

    other_path.rewind(0);
    cmd = other_path.vertex(&x, &y);
    while(!agg24::is_stop(cmd))
    {
        this->_has_curves |= agg24::is_curve(cmd);
        this->ptm.transform(&x,&y);
        vertices.add_vertex(x, y, cmd);
        cmd = other_path.vertex(&x, &y);
    }
    this->concat_ctm(other_path.ptm);
}
//{
//    agg24::conv_transform<agg24::path_storage> trans(p,ptm);
//    agg24::path_storage::add_path(trans);
//    concat_ctm(p.ptm);
//}

void compiled_path::lines(double* pts, int Npts)
{
    this->move_to(pts[0],pts[1]);
    for(int i=2; i < Npts*2; i+=2)
        this->line_to(pts[i],pts[i+1]);
}

void compiled_path::line_set(double* start, int Nstart, double* end, int Nend)
{
    int num_pts = (Nstart > Nend) ? Nend : Nstart;
    for (int i=0; i < num_pts*2; i += 2)
    {
        this->move_to(start[i], start[i+1]);
        this->line_to(end[i], end[i+1]);
    }
}

void compiled_path::rect(double x, double y, double sx, double sy)
{
    this->move_to(x, y);
    this->line_to(x, y+sy);
    this->line_to(x+sx, y+sy);
    this->line_to(x+sx, y);
    this->close_path();
}

void compiled_path::rect(kiva::rect_type &r)
{
    this->rect(r.x, r.y, r.w, r.h);
}

void compiled_path::rects(double* all_rects, int Nrects)
{
    double *tmp;
    for(int i = 0; i < Nrects*4; i+=4)
    {
        tmp = &all_rects[i];
        this->rect(tmp[0], tmp[1], tmp[2], tmp[3]);
    }
}

void compiled_path::rects(kiva::rect_list_type &rectlist)
{
    for (kiva::rect_list_type::iterator it=rectlist.begin(); it != rectlist.end(); it++)
    {
        this->rect(it->x, it->y, it->w, it->h);
    }
}

void compiled_path::_transform_ctm(agg24::trans_affine& m)
{
    this->ptm.premultiply(m);
}
void compiled_path::translate_ctm(double x, double y)
{
    agg24::trans_affine_translation m(x,y);
    this->_transform_ctm(m);
}

void compiled_path::rotate_ctm(double angle)
{
    agg24::trans_affine_rotation m(angle);
    this->_transform_ctm(m);
}

void compiled_path::scale_ctm(double sx, double sy)
{
    agg24::trans_affine_scaling m(sx,sy);
    this->_transform_ctm(m);
}

void compiled_path::concat_ctm(agg24::trans_affine& m)
{
    agg24::trans_affine m_copy(m);
    this->_transform_ctm(m_copy);
}

void compiled_path::set_ctm(agg24::trans_affine& m)
{
    this->ptm = agg24::trans_affine(m);
}

agg24::trans_affine compiled_path::get_ctm()
{
    return this->ptm;
}

void compiled_path::save_ctm()
{
    this->ptm_stack.push(this->ptm);
}

void compiled_path::restore_ctm()
{
    // !! need to check what error should be on empty stack.
    if ( !this->ptm_stack.empty())
    {
        this->ptm = this->ptm_stack.top();
        this->ptm_stack.pop();
    }
}
