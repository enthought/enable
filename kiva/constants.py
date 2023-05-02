# (C) Copyright 2005-2022 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
""" Constants used by core2d drawing engine. """

from enum import IntEnum, IntFlag

from numpy import array

# --------------------------------------------------------------------
# Line Dash Constants
# --------------------------------------------------------------------
NO_DASH = (0, array([0]))
del array

# --------------------------------------------------------------------
# Line Cap Constants
# --------------------------------------------------------------------

class LineCap(IntEnum):
    "Line cap styles."
    ROUND = 0
    BUTT = 1
    SQUARE = 2

CAP_ROUND = LineCap.ROUND
CAP_BUTT = LineCap.BUTT
CAP_SQUARE = LineCap.SQUARE

# --------------------------------------------------------------------
# Line Join Constants
# --------------------------------------------------------------------

class LineJoin(IntEnum):
    "Line join styles."
    ROUND = 0
    BEVEL = 1
    MITER = 2

JOIN_ROUND = LineJoin.ROUND
JOIN_BEVEL = LineJoin.BEVEL
JOIN_MITER = LineJoin.MITER

# --------------------------------------------------------------------
# Path Drawing Mode Constants
#
# Path drawing modes for path drawing methods.
# The values are chosen so that bit flags can be checked in a later
# C version.
# --------------------------------------------------------------------

class DrawMode(IntFlag):
    """Drawing mode flags.

    FILL
        Paint the path using the nonzero winding rule
        to determine the regions for painting.

    EOF_FILL
        Paint the path using the even-odd fill rule.

    STROKE
        Draw the outline of the path with the
        current width, end caps, etc settings.

    Note that FILL and EOF_FILL are mutually exclusive, so modes 3
    and 7 aren't valid.

    Outlines are stroked after any filling is performed.
    """
    FILL = 1
    EOF_FILL = 2
    STROKE = 4
    FILL_STROKE = 5
    EOF_FILL_STROKE = 6

FILL = DrawMode.FILL
EOF_FILL = DrawMode.EOF_FILL
STROKE = DrawMode.STROKE
FILL_STROKE = DrawMode.FILL_STROKE
EOF_FILL_STROKE = DrawMode.EOF_FILL_STROKE

# -----------------------------------------------------------------------------
# Font Constants
# -----------------------------------------------------------------------------

class FontStyle(IntFlag):
    """Basic font styles."""
    NORMAL = 0
    BOLD = 1
    ITALIC = 2

NORMAL = FontStyle.NORMAL
BOLD = FontStyle.BOLD
ITALIC = FontStyle.ITALIC
BOLD_ITALIC = FontStyle.BOLD | FontStyle.ITALIC

# convenience sets for styles
bold_styles = {BOLD, BOLD_ITALIC}
italic_styles = {ITALIC, BOLD_ITALIC}

# Font families, as defined by the Windows API, and their CSS equivalents
class FontFamily(IntEnum):
    """Standard font family names"""
    DEFAULT = 0
    SWISS = 1  # Sans-serif
    ROMAN = 2  # Serif
    MODERN = 3  # Monospace
    DECORATIVE = 4  # Fantasy
    SCRIPT = 5  # Cursive
    TELETYPE = 6

DEFAULT = FontFamily.DEFAULT
SWISS = FontFamily.SWISS
ROMAN = FontFamily.ROMAN
MODERN = FontFamily.MODERN
DECORATIVE = FontFamily.DECORATIVE
SCRIPT = FontFamily.SCRIPT
TELETYPE = FontFamily.TELETYPE


# Font weight constants
class FontWeight(IntEnum):
    """Font weights"""
    THIN = 100
    EXTRALIGHT = 200
    LIGHT = 300
    NORMAL = 400
    MEDIUM = 500
    SEMIBOLD = 600
    BOLD = 700
    EXTRABOLD = 800
    HEAVY = 900
    EXTRAHEAVY = 1000

WEIGHT_THIN = FontWeight.THIN
WEIGHT_EXTRALIGHT = FontWeight.EXTRALIGHT
WEIGHT_LIGHT = FontWeight.LIGHT
WEIGHT_NORMAL = FontWeight.NORMAL
WEIGHT_MEDIUM = FontWeight.MEDIUM
WEIGHT_SEMIBOLD = FontWeight.SEMIBOLD
WEIGHT_BOLD = FontWeight.BOLD
WEIGHT_EXTRABOLD = FontWeight.EXTRABOLD
WEIGHT_HEAVY = FontWeight.HEAVY
WEIGHT_EXTRAHEAVY = FontWeight.EXTRAHEAVY

# -----------------------------------------------------------------------------
# Text Drawing Mode Constants
# -----------------------------------------------------------------------------

class TextMode(IntEnum):
    """Text drawing mode.

    Determines how text is drawn to the screen.  If a CLIP flag is set, the
    font outline is added to the clipping path. Possible values:

    FILL
        fill the text

    STROKE
        paint the outline

    FILL_STROKE
        fill and outline

    INVISIBLE
        paint it invisibly ??

    FILL_CLIP
        fill and add outline clipping path

    STROKE_CLIP
        outline and add outline to clipping path

    FILL_STROKE_CLIP
        fill, outline, and add to clipping path

    CLIP
        add text outline to clipping path

    """
    FILL = 0
    STROKE = 1
    FILL_STROKE = 2
    INVISIBLE = 3
    FILL_CLIP = 4
    STROKE_CLIP = 5
    FILL_STROKE_CLIP = 6
    CLIP = 7
    OUTLINE = 8

TEXT_FILL = TextMode.FILL
TEXT_STROKE = TextMode.STROKE
TEXT_FILL_STROKE = TextMode.FILL_STROKE
TEXT_INVISIBLE = TextMode.INVISIBLE
TEXT_FILL_CLIP = TextMode.FILL_CLIP
TEXT_STROKE_CLIP = TextMode.STROKE_CLIP
TEXT_FILL_STROKE_CLIP = TextMode.FILL_STROKE_CLIP
TEXT_CLIP = TextMode.CLIP
TEXT_OUTLINE = TextMode.OUTLINE

# -----------------------------------------------------------------------------
# Subpath Drawing Primitive Constants
#
# Used by the drawing state machine to determine what object to draw.
# -----------------------------------------------------------------------------

class PathPrimitive(IntEnum):
    """Subpath drawing primitive constants

    Used by the drawing state machine to determine what object to draw.
    """
    POINT = 0
    LINE = 1
    LINES = 2
    RECT = 3
    CLOSE = 4
    CURVE_TO = 5
    QUAD_CURVE_TO = 6
    ARC = 7
    ARC_TO = 8

POINT = PathPrimitive.POINT
LINE = PathPrimitive.LINE
LINES = PathPrimitive.LINES
RECT = PathPrimitive.RECT
CLOSE = PathPrimitive.CLOSE
CURVE_TO = PathPrimitive.CURVE_TO
QUAD_CURVE_TO = PathPrimitive.QUAD_CURVE_TO
ARC = PathPrimitive.ARC
ARC_TO = PathPrimitive.ARC_TO


# -----------------------------------------------------------------------------
# Subpath CTM Constants
# -----------------------------------------------------------------------------

class CTM(IntEnum):
    """Subpath CTM Constants

    These are added so its possible for OpenGL to do the matrix transformations
    on the data (its much faster than doing it with NumPy).
    """
    SCALE = 5
    TRANSLATE = 6
    ROTATE = 7
    CONCAT = 8
    LOAD = 9

SCALE_CTM = CTM.SCALE
TRANSLATE_CTM = CTM.TRANSLATE
ROTATE_CTM = CTM.ROTATE
CONCAT_CTM = CTM.CONCAT
LOAD_CTM = CTM.LOAD


# -----------------------------------------------------------------------------
# Marker Types
#
# These are the marker types for draw_marker_at_points.  Some backends
# (like Agg) have fast implementations for these; other backends manually
# construct the paths representing these markers.
#
# Note that draw_marker_at_points takes a marker name as a string.
# -----------------------------------------------------------------------------

class Marker(IntEnum):
    """Marker Types

    These are the marker types for draw_marker_at_points.  Some backends
    (like Agg) have fast implementations for these; other backends manually
    construct the paths representing these markers.

    Note that draw_marker_at_points takes a marker name as a string.
    """
    NONE = 0
    SQUARE = 1
    DIAMOND = 2
    CIRCLE = 3
    CROSSED_CIRCLE = 4
    CROSS = 5
    TRIANGLE = 6
    INVERTED_TRIANGLE = 7
    PLUS = 8
    DOT = 9
    PIXEL = 10

NO_MARKER = Marker.NONE
SQUARE_MARKER = Marker.SQUARE
DIAMOND_MARKER = Marker.DIAMOND
CIRCLE_MARKER = Marker.CIRCLE
CROSSED_CIRCLE_MARKER = Marker.CROSSED_CIRCLE
CROSS_MARKER = Marker.CROSS
TRIANGLE_MARKER = Marker.TRIANGLE
INVERTED_TRIANGLE_MARKER = Marker.INVERTED_TRIANGLE
PLUS_MARKER = Marker.PLUS
DOT_MARKER = Marker.DOT
PIXEL_MARKER = Marker.PIXEL
