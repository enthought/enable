# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
"""

Affine Transformations
======================

- :func:`~.affine_identity`
- :func:`~.affine_from_values`
- :func:`~.affine_from_scale`
- :func:`~.affine_from_rotation`
- :func:`~.affine_from_translation`
- :func:`~.scale`
- :func:`~.rotate`
- :func:`~.translate`
- :func:`~.concat`
- :func:`~.invert`
- :func:`~.is_identity`
- :func:`~.affine_params`
- :func:`~.tsr_factor`
- :func:`~.trs_factor`
- :func:`~.transform_point`
- :func:`~.transform_points`

Constants
---------

- :attr:`~.IDENTITY` - represents the :func:`~.affine_identity` matrix.

Drawing Constants
=================

Line Dash Constants
-------------------

- :attr:`~.NO_DASH`

Line Cap Constants
------------------

- :attr:`~.CAP_ROUND`
- :attr:`~.CAP_BUTT`
- :attr:`~.CAP_SQUARE`

Line Join Constants
-------------------

- :attr:`~.JOIN_ROUND`
- :attr:`~.JOIN_BEVEL`
- :attr:`~.JOIN_MITER`

Path Drawing Mode Constants
---------------------------

- :attr:`~.FILL`
- :attr:`~.EOF_FILL`
- :attr:`~.STROKE`
- :attr:`~.FILL_STROKE`
- :attr:`~.EOF_FILL_STROKE`

Text Drawing Mode Constants
---------------------------

- :attr:`~.TEXT_FILL`
- :attr:`~.TEXT_STROKE`
- :attr:`~.TEXT_FILL_STROKE`
- :attr:`~.TEXT_INVISIBLE`
- :attr:`~.TEXT_FILL_CLIP`
- :attr:`~.TEXT_STROKE_CLIP`
- :attr:`~.TEXT_FILL_STROKE_CLIP`
- :attr:`~.TEXT_CLIP`
- :attr:`~.TEXT_OUTLINE`

Marker Types
------------

- :attr:`~.NO_MARKER`
- :attr:`~.SQUARE_MARKER`
- :attr:`~.DIAMOND_MARKER`
- :attr:`~.CIRCLE_MARKER`
- :attr:`~.CROSSED_CIRCLE_MARKER`
- :attr:`~.CROSS_MARKER`
- :attr:`~.TRIANGLE_MARKER`
- :attr:`~.INVERTED_TRIANGLE_MARKER`
- :attr:`~.PLUS_MARKER`
- :attr:`~.DOT_MARKER`
- :attr:`~.PIXEL_MARKER`

Fonts
=====

- :class:`~.Font`

Font Constants
--------------

Font Sizes

- :attr:`~.NORMAL`
- :attr:`~.BOLD`
- :attr:`~.ITALIC`
- :attr:`~.BOLD_ITALIC`

Font Families

- :attr:`~.DEFAULT`
- :attr:`~.SWISS`
- :attr:`~.ROMAN`
- :attr:`~.MODERN`
- :attr:`~.DECORATIVE`
- :attr:`~.SCRIPT`
- :attr:`~.TELETYPE`

Utilities
=========

- :func:`~.points_in_polygon`

"""
# flake8: noqa
from .affine import (
    affine_identity, affine_from_values, affine_from_scale,
    affine_from_rotation, affine_from_translation,
    scale, rotate, translate, concat, invert, is_identity, affine_params,
    tsr_factor, trs_factor, transform_point, transform_points, IDENTITY
)
from .constants import (
    NO_DASH, CAP_ROUND, CAP_BUTT, CAP_SQUARE,
    JOIN_ROUND, JOIN_BEVEL, JOIN_MITER,
    FILL, EOF_FILL, STROKE, FILL_STROKE, EOF_FILL_STROKE,
    NORMAL, BOLD, ITALIC, BOLD_ITALIC, DEFAULT,
    SWISS, ROMAN, MODERN, DECORATIVE, SCRIPT, TELETYPE,
    TEXT_FILL, TEXT_STROKE, TEXT_FILL_STROKE, TEXT_INVISIBLE, TEXT_FILL_CLIP,
    TEXT_STROKE_CLIP, TEXT_FILL_STROKE_CLIP, TEXT_CLIP, TEXT_OUTLINE,
    POINT, LINE, LINES, RECT, CLOSE, CURVE_TO, QUAD_CURVE_TO, ARC, ARC_TO,
    SCALE_CTM, TRANSLATE_CTM, ROTATE_CTM, CONCAT_CTM, LOAD_CTM,
    NO_MARKER, SQUARE_MARKER, DIAMOND_MARKER, CIRCLE_MARKER,
    CROSSED_CIRCLE_MARKER, CROSS_MARKER, TRIANGLE_MARKER,
    INVERTED_TRIANGLE_MARKER, PLUS_MARKER, DOT_MARKER, PIXEL_MARKER
)
from ._cython_speedups import points_in_polygon
from .fonttools import Font
