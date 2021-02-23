# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
""" Traits UI 'display only' SVG editor.
"""

# -----------------------------------------------------------------------------
#  Imports:
# -----------------------------------------------------------------------------

from enable.savage.trait_defs.ui.toolkit import toolkit_object

from traits.api import Property
from traitsui.api import BasicEditorFactory

# -----------------------------------------------------------------------------
#  'SVGEditor' editor factory class:
# -----------------------------------------------------------------------------


class SVGEditor(BasicEditorFactory):

    # The editor class to be created:
    klass = Property

    def _get_klass(self):
        """ Returns the toolkit-specific editor class to be instantiated.
        """
        return toolkit_object("svg_editor:SVGEditor")
