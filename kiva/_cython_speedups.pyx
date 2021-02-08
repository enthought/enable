# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
import numpy as np
from numpy cimport uint8_t
cimport _hit_test


def points_in_polygon(pts, poly_pts, use_winding=False):
    """Test whether point pairs in pts are within the polygon, poly_pts.

    Parameters
    ----------
    pts
        an Nx2 array of x,y point pairs (floating point).  Each point is tested
        to determine whether it falls within the polygon defined by `poly_pts`.
    poly_pts
        an Mx2 array of x,y point pairs (floating point) that define the
        boundaries of a polygon.  The last point is considered to be connected
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

        >>> import numpy as np
        >>> from enable import points_in_polygon
        >>> poly = np.array(((0.0, 0.0),
                            (10.0, 0.0),
                            (10.0, 10.0),
                            (0.0, 10.0)))
        >>> pts = np.array(((-1.0, -1.0),
                           (5.0, 5.0),
                           (15.0, 15.0)))
        >>> results = points_in_polygon(pts, poly)
        [0 1 0]

    """

    # Check the shape of pts and transpose if necessary.
    pts = np.asarray(pts, dtype=np.float64)
    if pts.size == 0:
        # Quick exit for empty pts array
        return np.zeros(0, dtype=np.uint8)
    if pts.ndim == 1:
        pts = np.reshape(pts, (1,) + np.shape(pts))
    if np.shape(pts)[1] != 2:
        if np.shape(pts)[0] == 2:
            pts = np.ascontiguousarray(np.transpose(pts))
        else:
            raise ValueError('pts must be an Nx2 or 2xN array')

    # Check the shape of poly_pts and transpose if necessary
    poly_pts = np.asarray(poly_pts, dtype=np.float64)
    if poly_pts.size == 0:
        # Quick exit for empty poly array
        return np.zeros(len(pts), dtype=np.uint8)
    if poly_pts.ndim == 1:
        poly_pts = np.reshape(poly_pts, (1,) + np.shape(poly_pts))
    if np.shape(poly_pts)[1] != 2:
        if np.shape(poly_pts)[0] == 2:
            poly_pts = np.ascontiguousarray(np.transpose(poly_pts))
        else:
            raise ValueError('poly_pts must be an Nx2 or 2xN array')

    cdef double[:, ::1] pts_view = np.ascontiguousarray(pts)
    cdef double[:, ::1] poly_pts_view = np.ascontiguousarray(poly_pts)
    cdef uint8_t[::1] results = np.zeros(len(pts), dtype=np.uint8)

    if use_winding:
        _hit_test.points_in_polygon_winding(&pts_view[0][0], pts_view.shape[0],
                                            &poly_pts_view[0][0],
                                            poly_pts_view.shape[0],
                                            &results[0], results.shape[0])
    else:
        _hit_test.points_in_polygon(&pts_view[0][0], pts_view.shape[0],
                                    &poly_pts_view[0][0],
                                    poly_pts_view.shape[0],
                                    &results[0], results.shape[0])

    return results.base.astype(np.bool_)
