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

from six import StringIO
from xml.etree.cElementTree import ElementTree

from enable.savage.svg.document import SVGDocument

from traitsui.qt4.editor import Editor

from pyface.qt import QtCore, QtSvg

#-------------------------------------------------------------------------------
#  'SVGEditor' class:
#-------------------------------------------------------------------------------

class SVGEditor(Editor):
    """ Traits UI 'display only' SVG editor.
    """

    scrollable = True

    #---------------------------------------------------------------------------
    #  Finishes initializing the editor by creating the underlying toolkit
    #  widget:
    #---------------------------------------------------------------------------

    def init(self, parent):
        """ Finishes initializing the editor by creating the underlying toolkit
            widget.
        """
        self.control = QtSvg.QSvgWidget()

    #---------------------------------------------------------------------------
    #  Updates the editor when the object trait changes external to the editor:
    #---------------------------------------------------------------------------

    def update_editor(self):
        """ Updates the editor when the object trait changes externally to the
            editor.
        """
        value = self.value

        if isinstance(value, SVGDocument):
            string_io = StringIO()
            ElementTree(value.tree).write(string_io)
            value = string_io.getvalue()

        self.control.load(QtCore.QByteArray(value))
