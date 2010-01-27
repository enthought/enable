#------------------------------------------------------------------------------
#
#  Copyright (c) 2009, Enthought, Inc.
#  All rights reserved.
# 
#  This software is provided without warranty under the terms of the BSD
#  license included in enthought/LICENSE.txt and may be redistributed only
#  under the conditions described in the aforementioned license.  The license
#  is also available online at http://www.enthought.com/licenses/BSD.txt
#
#  Thanks for using Enthought open source!
#  
#------------------------------------------------------------------------------

# ETS imports
from enthought.traits.api import Event

# Local imports
from svg_button_editor import SVGButtonEditor


class SVGButton ( Event ):
    """ Defines a trait whose UI editor is a button.
    """

    def __init__ ( self, label = '', filename = None,
                         tooltip = '', toggle=False,
                         toggle_state=False,
                         toggle_filename=None,
                         width = 32, height = 32,
                         orientation = 'vertical', width_padding = 4,
                         height_padding = 1, view = None, **metadata ):
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
            tooltip : string
                What to display when the mouse hovers over the button. An
                empty string implies no tooltip
            toggle : boolean
                Whether the button is a toggle with 2 states, or a regular
                button with 1 state

            Default Value
            -------------
            No default value because events do not store values.
        """

        self.editor = SVGButtonEditor( label          = label,
                                       filename       = filename,
                                       tooltip        = tooltip,
                                       toggle         = toggle,
                                       toggle_state   = toggle_state,
                                       toggle_filename   = toggle_filename,
                                       orientation    = orientation,
                                       width_padding  = width_padding,
                                       height_padding = height_padding,
                                       width          = width,
                                       height         = height,
                                       view           = view )

        super( SVGButton, self ).__init__( **metadata )
