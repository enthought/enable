# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
from kiva._marker_renderer import (
    MarkerRendererABGR32, MarkerRendererARGB32, MarkerRendererBGR24,
    MarkerRendererBGRA32, MarkerRendererRGB24, MarkerRendererRGBA32,
)

__all__ = ["MarkerRenderer"]

_renderers = {
    "abgr32": (MarkerRendererABGR32, 4),
    "argb32": (MarkerRendererARGB32, 4),
    "bgra32": (MarkerRendererBGRA32, 4),
    "rgba32": (MarkerRendererRGBA32, 4),
    "bgr24": (MarkerRendererBGR24, 3),
    "rgb24": (MarkerRendererRGB24, 3),
}


def MarkerRenderer(buffer, pix_format="bgra32", bottom_up=True):
    """ MarkerRenderer(buffer, pix_format="bgra32", bottom_up=True)
    Create a specialized renderer for implementing ``draw_marker_at_points``.

    Parameters
    ----------
    buffer : ndarray
        A MxNx{3,4} numpy array of uint8 to be used as the backing pixel store
    pix_format : str
        A string specifying the pixel format. Same as what it passed to
        ``GraphicsContext``.
    bottom_up : bool [optional, defaults to True]
        If True, the origin is bottom-left instead of top-left.

    Returns
    -------
    renderer : A new MarkerRenderer instance.
    """
    klass, components = _renderers.get(pix_format, (None, 0))
    if klass is None:
        raise ValueError(f"{pix_format} is not a supported pixel format")

    if (str(buffer.dtype) != "uint8" or buffer.ndim != 3
            or buffer.shape[2] != components):
        raise ValueError(f"Pixel buffer must be MxNx{components} and uint8")

    return klass(buffer, bottom_up=bottom_up)
