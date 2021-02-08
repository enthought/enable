# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!

# Enthought library imports
from traits.api import HasTraits, Int, Bool
from kiva.trait_defs.api import KivaFont
from enable.colors import ColorTrait


class TextFieldStyle(HasTraits):
    """ This class holds style settings for rendering an EnableTextField.
        fixme: See docstring on EnableBoxStyle
    """

    # The color of the text
    text_color = ColorTrait((0, 0, 0, 1.0))

    # The font for the text (must be monospaced!)
    font = KivaFont("Courier 12")

    # The color of highlighted text
    highlight_color = ColorTrait((0.65, 0, 0, 1.0))

    # The background color of highlighted items
    highlight_bgcolor = ColorTrait("lightgray")

    # The font for flagged text (must be monospaced!)
    highlight_font = KivaFont("Courier 14 bold")

    # The number of pixels between each line
    line_spacing = Int(3)

    # Space to offset text from the widget's border
    text_offset = Int(5)

    # Cursor properties
    cursor_color = ColorTrait((0, 0, 0, 1))
    cursor_width = Int(2)

    # Drawing properties
    border_visible = Bool(False)
    border_color = ColorTrait((0, 0, 0, 1))
    bgcolor = ColorTrait((1, 1, 1, 1))
