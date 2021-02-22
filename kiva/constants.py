# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
""" Constants used by core2d drawing engine. """

from numpy import array

# --------------------------------------------------------------------
# Line Dash Constants
# --------------------------------------------------------------------
NO_DASH = (0, array([0]))
del array

# --------------------------------------------------------------------
# Line Cap Constants
# --------------------------------------------------------------------

CAP_ROUND = 0
CAP_BUTT = 1
CAP_SQUARE = 2

# --------------------------------------------------------------------
# Line Join Constants
# --------------------------------------------------------------------

JOIN_ROUND = 0
JOIN_BEVEL = 1
JOIN_MITER = 2

# --------------------------------------------------------------------
# Path Drawing Mode Constants
#
# Path drawing modes for path drawing methods.
# The values are chosen so that bit flags can be checked in a later
# C version.
# --------------------------------------------------------------------

FILL = 1
EOF_FILL = 2
STROKE = 4
FILL_STROKE = 5
EOF_FILL_STROKE = 6

# -----------------------------------------------------------------------------
# Font Constants
# -----------------------------------------------------------------------------

NORMAL = 0
BOLD = 1
ITALIC = 2
BOLD_ITALIC = 3

# Font families, as defined by the Windows API, and their CSS equivalents
DEFAULT = 0
SWISS = 1  # Sans-serif
ROMAN = 2  # Serif
MODERN = 3  # Monospace
DECORATIVE = 4  # Fantasy
SCRIPT = 5  # Cursive
TELETYPE = 6

# -----------------------------------------------------------------------------
# Text Drawing Mode Constants
# -----------------------------------------------------------------------------

TEXT_FILL = 0
TEXT_STROKE = 1
TEXT_FILL_STROKE = 2
TEXT_INVISIBLE = 3
TEXT_FILL_CLIP = 4
TEXT_STROKE_CLIP = 5
TEXT_FILL_STROKE_CLIP = 6
TEXT_CLIP = 7
TEXT_OUTLINE = 8

# -----------------------------------------------------------------------------
# Subpath Drawing Primitive Constants
#
# Used by the drawing state machine to determine what object to draw.
# -----------------------------------------------------------------------------

POINT = 0
LINE = 1
LINES = 2
RECT = 3
CLOSE = 4
CURVE_TO = 5
QUAD_CURVE_TO = 6
ARC = 7
ARC_TO = 8


# -----------------------------------------------------------------------------
# Subpath CTM Constants
#
# These are added so its possible for OpenGL to do the matrix transformations
# on the data (its much faster than doing it with Numeric).
# -----------------------------------------------------------------------------

SCALE_CTM = 5
TRANSLATE_CTM = 6
ROTATE_CTM = 7
CONCAT_CTM = 8
LOAD_CTM = 9


# -----------------------------------------------------------------------------
# Marker Types
#
# These are the marker types for draw_marker_at_points.  Some backends
# (like Agg) have fast implementations for these; other backends manually
# construct the paths representing these markers.
#
# Note that draw_marker_at_points takes a marker name as a string.
# -----------------------------------------------------------------------------

NO_MARKER = 0
SQUARE_MARKER = 1
DIAMOND_MARKER = 2
CIRCLE_MARKER = 3
CROSSED_CIRCLE_MARKER = 4
CROSS_MARKER = 5
TRIANGLE_MARKER = 6
INVERTED_TRIANGLE_MARKER = 7
PLUS_MARKER = 8
DOT_MARKER = 9
PIXEL_MARKER = 10
