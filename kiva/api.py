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
