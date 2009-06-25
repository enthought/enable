#-------------------------------------------------------------------------------
#
#  Copyright (c) 2008, Enthought, Inc.
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
#  Date:   06/24/2000
#
#-------------------------------------------------------------------------------

""" Traits UI button editor for SVG images.
"""

#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

from enthought.traits.ui.qt4.editor import Editor

from PyQt4 import QtCore, QtGui

#-------------------------------------------------------------------------------
#  'SVGButtonEditor' class:
#-------------------------------------------------------------------------------
                               
class SVGButtonEditor(Editor):

    #---------------------------------------------------------------------------
    #  Editor interface
    #---------------------------------------------------------------------------
        
    def init(self, parent):
        """ Finishes initializing the editor by creating the underlying toolkit
            widget.
        """
        self.control = QtGui.QPushButton(self.factory.label)
        self.control.setFlat(True)
        self.control.setIcon(QtGui.QIcon(self.factory.filename))
        QtCore.QObject.connect(self.control, QtCore.SIGNAL('clicked()'),
                               self.update_object)
        if self.factory.tooltip:
            self.control.setToolTip(self.factory.tooltip)
        else:
            self.set_tooltip()

    def prepare(self, parent):
        """ Finishes setting up the editor. This differs from the base class
            in that self.update_editor() is not called at the end, which
            would fire an event.
        """
        name = self.extended_name
        if name != 'None':
            self.context_object.on_trait_change(self._update_editor, name,
                                                dispatch = 'ui')
        self.init(parent)
        self._sync_values()

    def update_object (self):
        """ Handles the user clicking the button by setting the factory value
            on the object.
        """
        self.value = self.factory.value

    def update_editor(self):
        """ Updates the editor when the object trait changes externally to the
            editor.
        """
        pass
