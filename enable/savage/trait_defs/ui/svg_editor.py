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
#  Author: Evan Patterson
#  Date:   06/24/2009
#
#------------------------------------------------------------------------------

""" Traits UI 'display only' SVG editor.
"""

#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

from enable.savage.trait_defs.ui.toolkit import toolkit_object

from traits.api import Property

from traitsui.basic_editor_factory import BasicEditorFactory

#-------------------------------------------------------------------------------
#  'SVGEditor' editor factory class:
#-------------------------------------------------------------------------------

class SVGEditor(BasicEditorFactory):

    # The editor class to be created:
    klass = Property

    def _get_klass(self):
        """ Returns the toolkit-specific editor class to be instantiated.
        """
        return toolkit_object('svg_editor:SVGEditor')
