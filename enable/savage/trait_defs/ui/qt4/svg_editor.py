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

from io import BytesIO
from xml.etree.cElementTree import ElementTree

from enable.savage.svg.document import SVGDocument

from traitsui.qt4.editor import Editor

from pyface.qt import QtCore, QtSvg

# -----------------------------------------------------------------------------
#  'SVGEditor' class:
# -----------------------------------------------------------------------------


class SVGEditor(Editor):
    """ Traits UI 'display only' SVG editor.
    """

    scrollable = True

    # -------------------------------------------------------------------------
    #  Finishes initializing the editor by creating the underlying toolkit
    #  widget:
    # -------------------------------------------------------------------------

    def init(self, parent):
        """ Finishes initializing the editor by creating the underlying toolkit
            widget.
        """
        self.control = QtSvg.QSvgWidget()

    # -------------------------------------------------------------------------
    #  Updates the editor when the object trait changes external to the editor:
    # -------------------------------------------------------------------------

    def update_editor(self):
        """ Updates the editor when the object trait changes externally to the
            editor.
        """
        value = self.value

        if isinstance(value, SVGDocument):
            bytes_io = BytesIO()
            ElementTree(value.tree).write(bytes_io)
            value = bytes_io.getvalue()

        self.control.load(QtCore.QByteArray(value))
