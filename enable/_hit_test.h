#ifndef ENABLE_HIT_TEST_H
#define ENABLE_HIT_TEST_H

namespace enable
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
