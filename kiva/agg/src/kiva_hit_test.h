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
