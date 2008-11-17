from enthought.traits.api import Event
from enthought.savage.traits.ui.wx.svg_button_editor import SVGButtonEditor

class SVGButton ( Event ):
    """ Defines a trait whose UI editor is a button.
    """

    def __init__ ( self, label = '', filename = None,
                         width = 32, height = 32,
                         orientation = 'vertical', width_padding = 7,
                         height_padding = 5, view = None, **metadata ):
        """ Returns a trait event whose editor is a button.

            Parameters
            ----------
            label : string
                The label for the button
            image : enthought.pyface.ImageResource
                An image to display on the button
            style : one of: 'button', 'radio', 'toolbar', 'checkbox'
                The style of button to display
            orientation : one of: 'horizontal', 'vertical'
                The orientation of the label relative to the image
            width_padding : integer between 0 and 31
                Extra padding (in pixels) added to the left and right sides of
                the button
            height_padding : integer between 0 and 31
                Extra padding (in pixels) added to the top and bottom of the
                button

            Default Value
            -------------
            No default value because events do not store values.
        """

        self.editor = SVGButtonEditor( label       = label,
                                    filename       = filename,
                                    orientation    = orientation,
                                    width_padding  = width_padding,
                                    height_padding = height_padding,
                                    width          = width,
                                    height         = height,
                                    view           = view )

        super( SVGButton, self ).__init__( **metadata )
