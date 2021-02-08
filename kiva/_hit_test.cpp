// (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
// All rights reserved.
//
// This software is provided without warranty under the terms of the BSD
// license included in LICENSE.txt and may be redistributed only under
// the conditions described in the aforementioned license. The license
// is also available online at http://www.enthought.com/licenses/BSD.txt
//
// Thanks for using Enthought open source!
#include "_hit_test.h"

namespace kiva
{
    // Adapted from: http://www.alienryderflex.com/polygon/
    //
    // The function will return TRUE if the point x,y is inside the
    // polygon, or FALSE if it is not. If the point x,y is exactly on
    // the edge of the polygon, then the function may return TRUE or
    // FALSE.
    //
    // Note that division by zero is avoided because the division is
    // protected by the "if" clause which surrounds it.
    //
    // Note: If the test point is on the border of the polygon, this
    // algorithm will deliver unpredictable results; i.e. the result
    // may be "inside" or "outside" depending on arbitrary factors
    // such as how the polygon is oriented with respect to the
    // coordinate system.

    inline bool toggle_odd_node(double x, double y,
                                double p1x, double p1y,
                                double p2x, double p2y)
    {
        bool toggle = false;
        if ( ((p1y<y) && (p2y>=y))
          || ((p2y<y) && (p1y>=y)) )
        {
            if (p1x + (y-p1y)/(p2y-p1y) * (p2x-p1x) < x)
            {
                toggle = true;
            }
        }
        return toggle;
    }

    bool point_in_polygon(double x, double y, double* poly_pts, int Npoly_pts)
    {

        bool odd_nodes=false;
        double p1_x, p1_y, p2_x, p2_y;

        for (int i=0; i<Npoly_pts-1; i++)
        {
            int ii = i*2;
            p1_x = poly_pts[ii];
            p1_y = poly_pts[ii+1];
            p2_x = poly_pts[ii+2];
            p2_y = poly_pts[ii+3];
            if (toggle_odd_node(x, y, p1_x, p1_y, p2_x, p2_y))
                odd_nodes =! odd_nodes;
        }

        // last point wraps back to beginning.
        p1_x = poly_pts[(Npoly_pts-1)*2];
        p1_y = poly_pts[(Npoly_pts-1)*2+1];
        p2_x = poly_pts[0];
        p2_y = poly_pts[1];
        if (toggle_odd_node(x, y, p1_x, p1_y, p2_x, p2_y))
            odd_nodes =! odd_nodes;

        return odd_nodes;
    }

    void points_in_polygon(double* pts, int Npts,
                           double* poly_pts, int Npoly_pts,
                           unsigned char* results, int Nresults)
    {
        // Nresults and Npts should match.

        for (int i=0; i < Npts; i++)
        {
            int ii = i*2;
            double x = pts[ii];
            double y = pts[ii+1];
            results[i] = point_in_polygon(x, y, poly_pts, Npoly_pts);
        }
    }

    inline double is_left(double x, double y,
                          double x1, double y1,
                          double x2, double y2)
    {
        return ( (x2 - x1) * (y - y1) - (x - x1) * (y2 - y1) );
    }

    inline int winding_increment(double x, double y,
                                 double x1, double y1,
                                 double x2, double y2)
    {
        if (y1 <= y)
        {
            if (y2 > y)
            {
                if ( is_left( x, y, x1, y1, x2, y2 ) > 0 )
                {
                    return 1;
                }
            }
        }
        else
        {
            if (y2 <= y)
            {
                if ( is_left( x, y, x1, y1, x2, y2 ) < 0 )
                {
                    return -1;
                }
            }
        }
        return 0;
    }


    bool point_in_polygon_winding(double x, double y,
                                  double* poly_pts, int Npoly_pts)
    {
        int winding_number = 0;
        double p1_x, p1_y, p2_x, p2_y;

        for (int i=0; i<Npoly_pts-1; i++) {
            int ii = i*2;
            p1_x = poly_pts[ii];
            p1_y = poly_pts[ii+1];
            p2_x = poly_pts[ii+2];
            p2_y = poly_pts[ii+3];

            winding_number += winding_increment(x, y, p1_x, p1_y, p2_x, p2_y);
        }

        // Last point wraps to the beginning.
        p1_x = poly_pts[(Npoly_pts-1)*2];
        p1_y = poly_pts[(Npoly_pts-1)*2+1];
        p2_x = poly_pts[0];
        p2_y = poly_pts[1];

        winding_number += winding_increment(x, y, p1_x, p1_y, p2_x, p2_y);

        return winding_number != 0;
    }

    void points_in_polygon_winding(double* pts, int Npts,
                                   double* poly_pts, int Npoly_pts,
                                   unsigned char* results, int Nresults)
    {
        // Nresults and Npts should match.

        for (int i=0; i < Npts; i++)
        {
            int ii = i*2;
            double x = pts[ii];
            double y = pts[ii+1];
            results[i] = point_in_polygon_winding(x, y, poly_pts, Npoly_pts);
        }
    }
}
