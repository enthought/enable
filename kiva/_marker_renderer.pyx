# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
import cython
import numpy as np
from numpy cimport uint8_t

cimport _marker_renderer

ctypedef _marker_renderer.marker_renderer_base renderer_base_t

@cython.internal
cdef class MarkerRendererBase:
    cdef renderer_base_t* _this
    cdef object py_array

    def __dealloc__(self):
        del self._this

    cdef int base_init(self, image) except -1:
        if image is None:
            raise ValueError('image argument must not be None.')

        # Retain a reference to the memory view supplied to the constructor
        # so that it lives as long as this object
        self.py_array = image

    def draw_markers(self, points, size, marker, fill, stroke):
        """draw_markers(points, size, marker, fill, stroke)
        Draw markers at a collection of points.

        :param points: An Nx2 iterable of (x, y) points for marker positions
        :param size: An integer pixel size for each marker
        :param marker: A Kiva marker enum integer
        :param fill: Fill color given as an iterable of 4 numbers (R, G, B, A)
        :param stroke: Line color given as an iterable of 4 numbers (R, G, B, A)

        :returns: True if any markers were drawn, False otherwise
        """
        cdef:
            double[:,::1] _points = np.asarray(points, dtype=np.float64, order='c')
            double[::1] _fill = np.asarray(fill, dtype=np.float64, order='c')
            double[::1] _stroke = np.asarray(stroke, dtype=np.float64, order='c')
            unsigned _size = <unsigned>size
            _marker_renderer.marker_type _marker = <_marker_renderer.marker_type>marker

        if _points.shape[1] != 2:
            msg = "points argument must be an iterable of (x, y) pairs."
            raise ValueError(msg)
        if _stroke.shape[0] != 4:
            msg = "stroke argument must be an iterable of 4 numbers."
            raise ValueError(msg)
        if _fill.shape[0] != 4:
            msg = "fill argument must be an iterable of 4 numbers."
            raise ValueError(msg)

        return self._this.draw_markers(
            &_points[0][0], _points.shape[0], _size, _marker,
            &_fill[0], &_stroke[0]
        )

    def transform(self, sx, sy, shx, shy, tx, ty):
        """transform(sx, sy, shx, shy, tx, ty)
        Set the transform to be applied to the marker points and size.

        :param sx: Scale in X
        :param sy: Scale in Y
        :param shx: Shear in X
        :param shy: Shear in Y
        :param tx: Translation in X
        :param ty: Translation in Y
        """
        cdef:
            double _sx = <double>sx
            double _sy = <double>sy
            double _shx = <double>shx
            double _shy = <double>shy
            double _tx = <double>tx
            double _ty = <double>ty

        self._this.transform(_sx, _sy, _shx, _shy, _tx, _ty)


# Template specializations
ctypedef _marker_renderer.marker_renderer[_marker_renderer.pixfmt_abgr32] renderer_abgr32_t
ctypedef _marker_renderer.marker_renderer[_marker_renderer.pixfmt_argb32] renderer_argb32_t
ctypedef _marker_renderer.marker_renderer[_marker_renderer.pixfmt_bgra32] renderer_bgra32_t
ctypedef _marker_renderer.marker_renderer[_marker_renderer.pixfmt_rgba32] renderer_rgba32_t
ctypedef _marker_renderer.marker_renderer[_marker_renderer.pixfmt_bgr24] renderer_bgr24_t
ctypedef _marker_renderer.marker_renderer[_marker_renderer.pixfmt_rgb24] renderer_rgb24_t

cdef class MarkerRendererABGR32(MarkerRendererBase):
    def __cinit__(self, uint8_t[:,:,::1] image, bottom_up=True):
        self.base_init(image)
        self._this = <renderer_base_t*> new renderer_abgr32_t(
            &image[0][0][0], image.shape[1], image.shape[0], image.strides[0], bottom_up
        )

cdef class MarkerRendererARGB32(MarkerRendererBase):
    def __cinit__(self, uint8_t[:,:,::1] image, bottom_up=True):
        self.base_init(image)
        self._this = <renderer_base_t*> new renderer_argb32_t(
            &image[0][0][0], image.shape[1], image.shape[0], image.strides[0], bottom_up
        )

cdef class MarkerRendererBGRA32(MarkerRendererBase):
    def __cinit__(self, uint8_t[:,:,::1] image, bottom_up=True):
        self.base_init(image)
        self._this = <renderer_base_t*> new renderer_bgra32_t(
            &image[0][0][0], image.shape[1], image.shape[0], image.strides[0], bottom_up
        )

cdef class MarkerRendererRGBA32(MarkerRendererBase):
    def __cinit__(self, uint8_t[:,:,::1] image, bottom_up=True):
        self.base_init(image)
        self._this = <renderer_base_t*> new renderer_rgba32_t(
            &image[0][0][0], image.shape[1], image.shape[0], image.strides[0], bottom_up
        )

cdef class MarkerRendererBGR24(MarkerRendererBase):
    def __cinit__(self, uint8_t[:,:,::1] image, bottom_up=True):
        self.base_init(image)
        self._this = <renderer_base_t*> new renderer_bgr24_t(
            &image[0][0][0], image.shape[1], image.shape[0], image.strides[0], bottom_up
        )

cdef class MarkerRendererRGB24(MarkerRendererBase):
    def __cinit__(self, uint8_t[:,:,::1] image, bottom_up=True):
        self.base_init(image)
        self._this = <renderer_base_t*> new renderer_rgb24_t(
            &image[0][0][0], image.shape[1], image.shape[0], image.strides[0], bottom_up
        )
