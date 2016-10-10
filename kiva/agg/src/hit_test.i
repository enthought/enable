%{
#include "kiva_hit_test.h"

%}

%include "agg_typemaps.i"

%apply (double* point_array, int point_count) {(double* pts, int Npts)};
%apply (double* point_array, int point_count) {(double* poly_pts, int Npoly_pts)};
%apply (unsigned char* results, int Nresults) {(unsigned char* results, int Nresults)};

namespace kiva
{
    bool point_in_polygon(double x, double y, double* poly_pts, int Npoly_pts);
    void points_in_polygon(double* pts, int Npts, 
                          double* poly_pts, int Npoly_pts,
                          unsigned char* results, int Nresults);
    bool point_in_polygon_winding(double x, double y, double* poly_pts, int Npoly_pts);
    void points_in_polygon_winding(double* pts, int Npts, 
                          double* poly_pts, int Npoly_pts,
                          unsigned char* results, int Nresults);
}

%pythoncode
%{
from numpy import asarray, shape, transpose, zeros, reshape, int32

def points_in_polygon(pts, poly_pts, use_winding=False):
    """ Test whether point pairs in pts are within the polygon, poly_pts.

    Parameters
    ----------
    pts 
        an Nx2 array of x,y point pairs (floating point).  Each point is tested
        to determine whether it falls within the polygon defined by `poly_pts`.                
    poly_pts
        an Mx2 array of x,y point pairs (floating point) that define the 
        boundaries of a polygon. The last point is considered to be connected
        to the first point.
    return 
        a 1D array of integers.  1 is returned if the corresponding x,y pair 
        in `pts` falls within `poly_pts`.  0 is returned otherwise.
        
    This algorithm works for complex polygons.  
         
    Note: If the test point is on the border of the polygon, this 
    algorithm will deliver unpredictable results; i.e. the result 
    may be "inside" or "outside" depending on arbitrary factors 
    such as how the polygon is oriented with respect to the 
    coordinate system.
        
    Adapted from: http://www.alienryderflex.com/polygon/
    
    Example::
    
        >>> from numpy import *
        >>> from kiva import agg        
        >>> poly = array(((0.0,   0.0),
                          (10.0,  0.0),
                          (10.0, 10.0),
                          ( 0.0, 10.0)))                
        >>> pts = array(((-1.0, -1.0),
                         ( 5.0,  5.0),  
                         ( 15.0, 15.0)))
        >>> results = agg.points_in_polygon(pts, poly)
        [0 1 0]
        
        
    """
    
    # Check the shape of pts and transpose if necessary.
    pts = asarray(pts)
    if pts.ndim == 1:
        pts = reshape(pts, (1,)+shape(pts))
    if shape(pts)[1] != 2:
        if shape(pts)[0] == 2:
            pts = transpose(pts)
        else:
            raise ValueError('pts must be an Nx2 or 2xN array')

    # Check the shape of poly_pts and transpose if necessary
    poly_pts = asarray(poly_pts)
    if poly_pts.ndim == 1:
        poly_pts = reshape(poly_pts, (1,)+shape(poly_pts))
    if shape(poly_pts)[1] != 2:
        if shape(poly_pts)[0] == 2:
            poly_pts = transpose(poly_pts)
        else:
            raise ValueError('poly_pts must be an Nx2 or 2xN array')

    results = zeros(len(pts), bool)
    if use_winding:
        _agg.points_in_polygon_winding(pts, poly_pts, results)
    else:
        _agg.points_in_polygon(pts, poly_pts, results)
    return results    
%}
