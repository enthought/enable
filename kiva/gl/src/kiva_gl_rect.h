// (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
// All rights reserved.
//
// This software is provided without warranty under the terms of the BSD
// license included in LICENSE.txt and may be redistributed only under
// the conditions described in the aforementioned license. The license
// is also available online at http://www.enthought.com/licenses/BSD.txt
//
// Thanks for using Enthought open source!
#ifndef KIVA_GL_RECT_H
#define KIVA_GL_RECT_H

#include <vector>

#include "agg_basics.h"
#include "kiva_gl_basics.h"

namespace kiva_gl
{

    //-----------------------------------------------------------------------
    // graphics_state class
    //-----------------------------------------------------------------------

    class rect_type {
    public:
        // constructors
        inline rect_type(): x(0), y(0), w(-1), h(-1) { }
        inline rect_type(double newx, double newy, double neww, double newh)
                        : x(newx), y(newy), w(neww), h(newh) { }
        inline rect_type(kiva_gl_agg::rect_i r) { *this = r; }
        inline rect_type(kiva_gl_agg::rect_d r) { *this = r; }

        // conversion from kiva_gl_agg::rect
        inline rect_type& operator=(kiva_gl_agg::rect_i &r)
        {
            x = int(r.x1);
            y = int(r.y1);
            w = int(r.x2 - r.x1);
            h = int(r.y2 - r.y1);
            return *this;
        }

        inline rect_type& operator=(kiva_gl_agg::rect_d &r)
        {
            x = r.x1;
            y = r.y1;
            w = r.x2 - r.x1;
            h = r.y2 - r.y1;
            return *this;
        }

        inline bool operator==(rect_type& other)
        {
            return ((x == other.x) && (y == other.y) && (w == other.w) && (h == other.h));
        }

        inline bool operator!=(rect_type& other)
        {
            return !(*this == other);
        }

        // conversion to kiva_gl_agg::rect
        inline operator kiva_gl_agg::rect_i() const { return kiva_gl_agg::rect_i(int(x), int(y), int(w), int(h)); }
        inline operator kiva_gl_agg::rect_d() const { return kiva_gl_agg::rect_d(x, y, w, h); }

        // conversion to double[4]
        inline double operator[](unsigned int ndx) const
        {
            switch (ndx)
            {
            case 0: return x;
            case 1: return y;
            case 2: return w;
            case 3: return h;
            }
        }

        // comparison
        inline bool operator==(const rect_type &b) const
        {
            return ((this->x == b.x) && (this->y == b.y) && (this->w == b.w) && (this->h == b.h));
        }

        // utility functions:
        inline double x2() const { return x+w; }
        inline double y2() const { return y+h; }

        double x, y, w, h;
    };

    typedef std::vector<rect_type> rect_list_type;
    typedef rect_list_type::iterator rect_iterator;

    // This returns the rectangle representing the overlapping area between
    // rectangles a and b.  If they do not overlap, the returned rectangle
    // will have width and height -1.
    //
    // (We use -1 instead of 0 because Agg will accept clip rectangles of
    // size 0.)
    rect_type disjoint_intersect(const rect_type &a, const rect_type &b);

    // Returns a list of rectangles resulting from the intersection of the
    // input list of rectangles.  If there are no intersection regions,
    // returns an empty list.
    rect_list_type disjoint_intersect(const rect_list_type &rects);

    // Intersects a single rectangle against a list of existing, non-
    // intersecting rectangles, and returns a list of the intersection regions.
    // If there are no intersection regions, returns an empty list.
    rect_list_type disjoint_intersect(const rect_list_type &original_list,
                                      const rect_type &new_rect);

    rect_list_type disjoint_union(const rect_type &a, const rect_type &b);
    rect_list_type disjoint_union(const rect_list_type &rects);
    rect_list_type disjoint_union(rect_list_type original_list,
                                  const rect_type &new_rect);

}

#endif
