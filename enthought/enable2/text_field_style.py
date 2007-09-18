# Enthought library imports
from enthought.traits.api import HasTraits, Int, Bool
from enthought.kiva.traits.api import KivaFont
from enthought.enable2.traits.api import RGBAColor
from enthought.enable2.colors import ColorTrait


class TextFieldStyle(HasTraits):
    """ This class holds style settings for rendering an EnableTextField.
        fixme: See docstring on EnableBoxStyle
    """

    # The color of the text
    text_color = RGBAColor((0,0,0,1))

    # The color of highlighted text
    highlight_color = ColorTrait("red")

    # The background color of highlighted items
    highlight_bgcolor = ColorTrait("lightgray")
    
    # The font for the text (must be monospaced!)
    font = KivaFont("Andale Mono 15")

    # The number of pixels between each line
    line_spacing = Int(3)

    # Space to offset text from the widget's border
    text_offset = Int(5)

    # Cursor properties
    cursor_color = RGBAColor((0,0,0,1))
    cursor_width = Int(2)

    # Drawing properties
    border_visible = Bool(True)
    border_color = RGBAColor((0,0,0,1))
    bgcolor = RGBAColor((1,1,1,1))
