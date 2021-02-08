# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
""" Traits UI button editor for SVG images.
"""

# -----------------------------------------------------------------------------
#  Imports:
# -----------------------------------------------------------------------------

from enable.savage.trait_defs.ui.toolkit import toolkit_object

from traits.api import Bool, Enum, Int, Property, Range, Str, Any

from traitsui.api import BasicEditorFactory, View
from traitsui.ui_traits import AView

# -----------------------------------------------------------------------------
#  'SVGEditor' editor factory class:
# -----------------------------------------------------------------------------


class SVGButtonEditor(BasicEditorFactory):

    # The editor class to be created
    klass = Property

    # Value to set when the button is clicked
    value = Property

    label = Str

    filename = Str

    # Extra padding to add to both the left and the right sides
    width_padding = Range(0, 31, 3)

    # Extra padding to add to both the top and the bottom sides
    height_padding = Range(0, 31, 3)

    # Presentation style
    style = Enum("button", "radio", "toolbar", "checkbox")

    # Orientation of the text relative to the image
    orientation = Enum("vertical", "horizontal")

    # The optional view to display when the button is clicked:
    view = AView

    width = Int(32)
    height = Int(32)

    tooltip = Str

    toggle = Bool(True)

    # the toggle state displayed
    toggle_state = Bool(False)

    # a file holding the image to display when toggled
    toggle_filename = Any

    toggle_label = Str

    toggle_tooltip = Str

    traits_view = View(["value", "|[]"])

    # -------------------------------------------------------------------------
    #  object API
    # -------------------------------------------------------------------------

    def __init__(self, **traits):
        self._value = 0
        super(SVGButtonEditor, self).__init__(**traits)

    # -------------------------------------------------------------------------
    #  Traits properties
    # -------------------------------------------------------------------------

    def _get_klass(self):
        """ Returns the toolkit-specific editor class to be instantiated.
        """
        return toolkit_object("svg_button_editor:SVGButtonEditor")

    def _get_value(self):
        return self._value

    def _set_value(self, value):
        self._value = value
        if isinstance(value, str):
            try:
                self._value = int(value)
            except ValueError:
                try:
                    self._value = float(value)
                except ValueError:
                    pass
