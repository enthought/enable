# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
from .agg import *

pix_format_string_map = {}
pix_format_string_map["gray8"] = pix_format_gray8
pix_format_string_map["rgb555"] = pix_format_rgb555
pix_format_string_map["rgb565"] = pix_format_rgb565
pix_format_string_map["rgb24"] = pix_format_rgb24
pix_format_string_map["bgr24"] = pix_format_bgr24
pix_format_string_map["rgba32"] = pix_format_rgba32
pix_format_string_map["argb32"] = pix_format_argb32
pix_format_string_map["abgr32"] = pix_format_abgr32
pix_format_string_map["bgra32"] = pix_format_bgra32

default_pix_format = "bgra32"

import types

try:
    # Define a system-dependent GraphicsContext if there is a PixelMap
    # class defined for the system (i.e. if plat_support was built)
    from .plat_support import PixelMap

    class GraphicsContextSystem(GraphicsContextArray):
        def __init__(self, size, pix_format=default_pix_format,
                     interpolation="nearest", base_pixel_scale=1.0,
                     bottom_up=True):
            assert isinstance(size, tuple), repr(size)
            width, height = size
            pixel_map = PixelMap(
                width,
                height,
                pix_format_string_map[pix_format],
                255,
                bool(bottom_up),
            ).set_bmp_array()
            GraphicsContextArray.__init__(
                self, pixel_map.bmp_array, pix_format, interpolation,
                base_pixel_scale, bottom_up
            )
            self.pixel_map = pixel_map


except ImportError as ex:
    # warn to stderr containing the exception. The warning should
    # be an ImportWarning, but that is python 2.5+ specific
    import warnings

    warnings.warn("Error initializing Agg: %s" % ex, Warning, 2)

    GraphicsContextSystem = None


def points_in_polygon(pts, poly_pts, use_winding=False):
    """Keep this function around for old code, but warn anyone who calls it.
    """
    import inspect
    import warnings
    from kiva.api import points_in_polygon as new_points_in_polygon

    msg = "points_in_polygon() has moved to kiva.api"
    frame = inspect.currentframe().f_back
    warnings.warn_explicit(
        msg,
        category=DeprecationWarning,
        filename=inspect.getfile(frame.f_code),
        lineno=frame.f_lineno,
    )

    new_points_in_polygon(pts, poly_pts, use_winding=use_winding)
