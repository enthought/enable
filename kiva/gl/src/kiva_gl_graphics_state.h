// (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
// All rights reserved.
//
// This software is provided without warranty under the terms of the BSD
// license included in LICENSE.txt and may be redistributed only under
// the conditions described in the aforementioned license. The license
// is also available online at http://www.enthought.com/licenses/BSD.txt
//
// Thanks for using Enthought open source!
#ifndef KIVA_GL_GRAPHICS_STATE_H
#define KIVA_GL_GRAPHICS_STATE_H

#include <vector>
#include "agg_trans_affine.h"

#include "kiva_gl_constants.h"
#include "kiva_gl_dash_type.h"
#include "kiva_gl_compiled_path.h"

#include <iostream>

namespace kiva_gl
{
    //-----------------------------------------------------------------------
    // graphics_state class
    //-----------------------------------------------------------------------

    class graphics_state
    {
        public:

            // line attributes
            kiva_gl_agg::rgba line_color;
            double line_width;
            kiva_gl::line_cap_e line_cap;
            kiva_gl::line_join_e line_join;
            kiva_gl::dash_type line_dash;

            // other attributes
            kiva_gl::blend_mode_e blend_mode;
            kiva_gl_agg::rgba fill_color;
            double alpha;

            // clipping path
            // In general, we need a path to store the clipping region.
            // However, in most cases, the clipping region can be represented
            // by a list of rectangles.  The graphics state can support one or
            // the other, but not both.  By default, device_space_clip_rects is
            // used; but as soon as a non-rectangular clip path is added to
            // the graphics state or the rectangular region is rotated, then
            // it becomes an arbitrary clipping path.
            //
            // device_space_clip_rects always contains at least one rectangle.
            // In the event that everything is clipped out, the clip rectangle
            // will have dimensions (0,0).
            //
            // The function use_rect_clipping is used to determine whether or
            // not to use device_space_clip_rects.  'true' means to use it, 'false'
            // means ot use clipping_path;
            kiva_gl::compiled_path clipping_path;
            std::vector<kiva_gl::rect_type> device_space_clip_rects;
            inline bool use_rect_clipping();

            double current_point[2]; // !! not sure about this.
            int should_antialias;
            double miter_limit;
            double flatness;  // !! not sure about this type.

            // double rendering_intent; // !! I know this type is wrong...

            graphics_state()
            : line_color(kiva_gl_agg::rgba(0.0, 0.0, 0.0))
            , line_width(1.0)
            , line_cap(kiva_gl::CAP_BUTT)
            , line_join(kiva_gl::JOIN_MITER)
            , blend_mode(kiva_gl::blend_normal)
            , fill_color(kiva_gl_agg::rgba(0.0, 0.0, 0.0))
            , alpha(1.0)
            , should_antialias(1)
            {
            }

            ~graphics_state()
            {
            }

            inline bool is_singleclip()
            {
                return (device_space_clip_rects.size() <= 1 ? true : false);
            }
    };

    inline bool graphics_state::use_rect_clipping()
    {
        if (clipping_path.total_vertices() > 0)
        {
            std::cout << "clipping path has vertices" << std::endl;
            return false;
        }

        return true;
    }

}

#endif
