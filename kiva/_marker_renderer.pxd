# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
from libcpp cimport bool

cdef extern from "marker_renderer.h" namespace "agg24markers":
    cdef cppclass pixfmt_abgr32:
        pass
    cdef cppclass pixfmt_argb32:
        pass
    cdef cppclass pixfmt_bgra32:
        pass
    cdef cppclass pixfmt_rgba32:
        pass
    cdef cppclass pixfmt_bgr24:
        pass
    cdef cppclass pixfmt_rgb24:
        pass


cdef extern from "marker_renderer.h" namespace "kiva_markers":
    # This is just here for the type signature
    cdef enum marker_type:
        pass

    # Abstract base class
    cdef cppclass marker_renderer_base:
        bool draw_markers(double* pts, unsigned Npts,
                          unsigned size, marker_type marker,
                          double* fill, double* stroke)
        void transform(double sx, double sy,
                       double shx, double shy,
                       double tx, double ty)

    # Template class
    cdef cppclass marker_renderer[pixfmt_T]:
        marker_renderer(unsigned char* buf, unsigned width, unsigned height,
                        int stride, bool bottom_up)
