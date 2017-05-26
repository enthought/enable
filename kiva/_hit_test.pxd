

cdef extern from "_hit_test.h" namespace "kiva":
    void points_in_polygon(double* pts, int Npts,
                          double* poly_pts, int Npoly_pts,
                          unsigned char* results, int Nresults)

    void points_in_polygon_winding(double* pts, int Npts,
                                   double* poly_pts, int Npoly_pts,
                                   unsigned char* results, int Nresults)
