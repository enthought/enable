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

from traitsui.wx.editor import Editor

from enable.savage.svg.backends.wx.renderer import Renderer as WxRenderer
from enable.savage.svg.backends.kiva.renderer import Renderer as KivaRenderer

from .kiva_render_panel import RenderPanel as KivaRenderPanel
from .wx_render_panel import RenderPanel as WxRenderPanel

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
        document = self.value

        # TODO: the document should not know about the renderer, this should
        # be an attribute of the editor

        if document.renderer == WxRenderer:
            self.control = WxRenderPanel(parent, document=document)
        else:
            self.control = KivaRenderPanel(parent, document=document)

    # -------------------------------------------------------------------------
    #  Updates the editor when the object trait changes external to the editor:
    # -------------------------------------------------------------------------

    def update_editor(self):
        """ Updates the editor when the object trait changes externally to the
            editor.
        """
        if self.control.document != self.value:
            self.control.document = self.value
            self.control.Refresh()
