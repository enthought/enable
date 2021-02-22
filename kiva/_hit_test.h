// (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
// All rights reserved.
//
// This software is provided without warranty under the terms of the BSD
// license included in LICENSE.txt and may be redistributed only under
// the conditions described in the aforementioned license. The license
// is also available online at http://www.enthought.com/licenses/BSD.txt
//
// Thanks for using Enthought open source!
#ifndef KIVA_HIT_TEST_H
#define KIVA_HIT_TEST_H

namespace kiva
{
    bool point_in_polygon(double x, double y, 
                          double* poly_pts, int Npoly_pts);
    void points_in_polygon(double* pts, int Npts, 
                          double* poly_pts, int Npoly_pts,
                          unsigned char* results, int Nresults);
    bool point_in_polygon_winding(double x, double y, 
                                  double* poly_pts, int Npoly_pts);
    void points_in_polygon_winding(double* pts, int Npts, 
                                   double* poly_pts, int Npoly_pts,
                                   unsigned char* results, int Nresults);

}
#endif
