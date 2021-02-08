// (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
// All rights reserved.
//
// This software is provided without warranty under the terms of the BSD
// license included in LICENSE.txt and may be redistributed only under
// the conditions described in the aforementioned license. The license
// is also available online at http://www.enthought.com/licenses/BSD.txt
//
// Thanks for using Enthought open source!
#ifndef KIVA_GL_COMPILED_PATH_H
#define KIVA_GL_COMPILED_PATH_H

#include <iostream>
#include <stack>

#include "agg_basics.h"
#include "agg_path_storage.h"
#include "agg_trans_affine.h"

#include "kiva_gl_rect.h"


namespace kiva_gl
{

    class compiled_path : public kiva_gl_agg::path_storage
    {

        /*-------------------------------------------------------------------
         This extends the standard kiva_gl_agg::path_storage class to include
         matrix transforms within the path definition.  Doing so requires
         overriding a number of methods to apply the matrix transformation
         to vertices added to the path.

         The overridden methods are:
             move_to
             line_to
             add_path
             curve3
             curve4
             add_path

         There are a few others that need to be looked at also...

         In addition, we need to add several methods:
             translate_ctm
             rotate_ctm
             scale_ctm
             concat_ctm
             set_ctm
             get_ctm
             save_ctm
             restore_ctm
        -------------------------------------------------------------------*/

        // hack to get VC++ 6.0 to compile correctly
        typedef kiva_gl_agg::path_storage base;

        /*-------------------------------------------------------------------
         ptm -- path transform matrix.

         This is used to transform each point added to the path.  It begins
         as an identity matrix and accumulates every transform made during the
         path formation.  At the end of the path creation, the ptm holds
         the total transformation seen during the path formation.  It
         can thus be multiplied with the ctm to determine what the ctm
         should be after the compiled_path has been drawn.

         Todo: Should this default to the identity matrix or the current ctm?
        -------------------------------------------------------------------*/
        kiva_gl_agg::trans_affine ptm;

        // ptm_stack is used for save/restore of the ptm
        std::stack<kiva_gl_agg::trans_affine> ptm_stack;

        // If the path contains curves, this value is true;
        bool _has_curves;

        public:
            // constructor
            compiled_path() : base(), ptm(kiva_gl_agg::trans_affine())
            {}

            //---------------------------------------------------------------
            // path_storage interface
            //---------------------------------------------------------------
            void remove_all();

            void begin_path();
            void close_path();
            void move_to(double x, double y);
            void line_to(double x, double y);
            void quad_curve_to(double x_ctrl, double y_ctrl,
                               double x_to, double y_to);
            void curve_to(double x_ctrl1, double y_ctrl1,
                          double x_ctrl2, double y_ctrl2,
                          double x_to, double y_to);

            // see graphics_context_base for descriptions of these functions
            void arc(double x, double y, double radius, double start_angle,
                     double end_angle, bool cw=false);
            void arc_to(double x1, double y1, double x2, double y2,
                        double radius);

            void add_path(compiled_path& other_path);
            void lines(double* pts, int Npts);
            void line_set(double* start, int Nstart, double* end, int Nend);
            void rect(double x, double y, double sx, double sy);
            void rect(kiva_gl::rect_type &rect);
            void rects(double* all_rects, int Nrects);
            void rects(kiva_gl::rect_list_type &rectlist);

            //---------------------------------------------------------------
            // compiled_path interface
            //---------------------------------------------------------------

            void _transform_ctm(kiva_gl_agg::trans_affine& m);
            void translate_ctm(double x, double y);
            void rotate_ctm(double angle);
            void scale_ctm(double sx, double sy);
            void concat_ctm(kiva_gl_agg::trans_affine& m);
            void set_ctm(kiva_gl_agg::trans_affine& m);
            kiva_gl_agg::trans_affine get_ctm();

            //---------------------------------------------------------------
            // save/restore ptm methods
            //---------------------------------------------------------------
            void save_ctm();
            void restore_ctm();

            //---------------------------------------------------------------
            // Test whether curves exist in path.
            //---------------------------------------------------------------
            inline bool has_curves() { return this->_has_curves;}
    };
}

#endif
