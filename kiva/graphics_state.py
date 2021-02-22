# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
""" GraphicsState Class

The GraphicsState class is used Kiva backends which need to have their state
tracked by Python, rather than by an internal graphics state (eg. Wx, SVG and
PDF backends, but not Agg or QPainter).
"""
import copy

from numpy import array, float64

from .constants import CAP_ROUND, JOIN_MITER, TEXT_FILL
from .fonttools import Font
from .line_state import LineState
import kiva.affine as affine


class GraphicsState(LineState):
    """ Holds information used by a graphics context when drawing.

    I'm not sure if these should be a separate class, a dictionary,
    or part of the GraphicsContext object.  Making them a dictionary
    or object simplifies save_state and restore_state a little bit.

    Also, this is a pretty good candidate for using slots.  I'm not
    going to use them right now, but, if we standardize on 2.2, slots might
    speed things up some.

    Attributes
    ----------

    ctm
        context transform matrix
    fill_color
        RGBA array(4) of values 0.0 to 1.0
    alpha
        transparency value of drawn objects
    font
        either a special device independent font
        object (what does anygui use?) or a
        device dependent font object.
    text_matrix
        coordinate transformation matrix for text
    clipping_path
        defines the path of the clipping region.
        For now, this can only be a rectangle.
    current_point
        location where next object is drawn.
    should_antialias
        whether anti-aliasing should be used when
        drawing lines and fonts
    miter_limit
        specifies when and when not to miter line joins.
    flatness
        specifies tolerance for bumpiness of curves
    character_spacing
        spacing between drawing text characters
    text_drawing_mode
        style for drawing text: outline, fill, etc.

    These are inherited from LineState:

    line_color
        RGBA array(4) of values 0.0 to 1.0
    line_width
        width of drawn lines
    line_join
        style of how lines are joined.  The choices
        are: JOIN_ROUND, JOIN_BEVEL, JOIN_MITER
    line_cap
        style of the end cap on lines.  The choices
        are: CAP_ROUND, CAP_SQUARE, CAP_BUTT
    line_dash
        (phase,pattern) dash pattern for lines.
        phase is a single value specifying how many
        units into the pattern to start.  dash is
        a 1-D array of floats that alternate between
        specifying the number of units on and off
        in the pattern.  When the end of the array
        is reached, the pattern repeats.

    Not yet supported:

    rendering_intent
        deals with colors and color correction in
        a sophisticated way.
    """

    def __init__(self):
        # Line state default values.
        line_color = array((0.0, 0.0, 0.0, 1.0))
        line_width = 1
        line_cap = CAP_ROUND
        line_join = JOIN_MITER
        line_dash = (0, array([0]))  # This will draw a solid line

        # FIXME: This is a very wierd class. The following code is here to
        # make the basecore2d and the PS, SVG context managers happy
        super(GraphicsState, self).__init__(
            line_color, line_width, line_cap, line_join, line_dash
        )
        self.line_state = self

        # All other default values.
        self.ctm = affine.affine_identity()
        self.fill_color = array((0.0, 0.0, 0.0, 1.0))
        self.alpha = 1.0
        self.font = Font()
        self.text_matrix = affine.affine_identity()
        self.clipping_path = None  # Not sure what the default should be?
        # Technically uninitialized in the PDF spec, but 0,0 seems fine to me:
        self.current_point = array((0, 0), dtype=float64)

        self.antialias = True
        self.miter_limit = 1.0
        self.flatness = 1.0
        self.character_spacing = 0.0
        self.text_drawing_mode = TEXT_FILL
        self.alpha = 1.0

    def copy(self):
        return copy.deepcopy(self)
