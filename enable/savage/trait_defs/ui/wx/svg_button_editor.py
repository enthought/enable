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

# Standard library imports
import copy
import sys
import xml.etree.cElementTree as etree
import os.path

# System library imports
import wx

# ETS imports
from enable.savage.svg.document import SVGDocument
from enable.savage.svg.backends.wx.renderer import Renderer
from traits.api import Instance
from traitsui.wx.constants import WindowColor
from traitsui.wx.editor import Editor

# Local imports
from .wx_render_panel import RenderPanel


class ButtonRenderPanel(RenderPanel):
    def __init__(self, parent, button, padding=(8, 8)):
        self.button = button
        self.document = button.document
        self.state = "up"

        self.toggle_document = button.toggle_document
        self.toggle_state = button.factory.toggle_state

        self.padding = padding

        super(ButtonRenderPanel, self).__init__(parent, document=self.document)

    def DoGetBestSize(self):
        label = self.button.factory.label
        if len(label):
            dc = wx.ScreenDC()
            dc.SetFont(wx.SystemSettings.GetFont(wx.SYS_DEFAULT_GUI_FONT))
            label_size = dc.GetTextExtent(label)
        else:
            label_size = (0, 0)
        width = max(self.button.factory.width, label_size[0])
        height = self.button.factory.height + label_size[1]
        return wx.Size(width + self.padding[0], height + self.padding[1])

    def GetBackgroundColour(self):
        bgcolor = copy.copy(WindowColor)
        if self.state == "down":
            red, green, blue = bgcolor.Get()[:3]
            red -= 15
            green -= 15
            blue -= 15
            bgcolor.Set(red, green, blue, 255)
        return bgcolor

    def OnPaint(self, evt):
        dc = wx.BufferedPaintDC(self)
        dc.SetBackground(wx.Brush(self.GetBackgroundColour()))
        dc.Clear()
        dc.SetFont(wx.SystemSettings.GetFont(wx.SYS_DEFAULT_GUI_FONT))

        gc = wx.GraphicsContext.Create(dc)

        if (self.toggle_state
                and self.button.factory.toggle
                and not self.button.factory.toggle_filename):
            self._draw_toggle(gc)

        # Put the icon in the middle of the alotted space. If the text is wider
        # than the icon, then the best_size will be wider, in which case we
        # want to put the icon in a little bit more towards the center,
        # otherwise, the icon will be drawn starting after the left padding.

        best_size = self.DoGetBestSize()
        x_offset = (best_size.width - self.button.factory.width) / 2.0
        y_offset = self.padding[1] / 2.0
        gc.Translate(x_offset, y_offset)
        gc.Scale(float(self.zoom_x) / 100, float(self.zoom_y) / 100)

        if (self.toggle_state
                and self.button.factory.toggle
                and self.button.factory.toggle_filename):
            self.toggle_document.render(gc)
            label_text = self.button.factory.toggle_label
        else:
            self.document.render(gc)
            label_text = self.button.factory.label

        # Reset the translation and zoom, then draw the text at an offset
        # based on the text width. There is a minor gotcha for supporting
        # multiple platforms here, Translate and DrawText behave differently
        # on different platforms.
        # It would be nice is a cross platform library actually worked the
        # same across platforms...

        text_width = dc.GetTextExtent(label_text)[0]
        text_x = (best_size.width - text_width) / 2.0
        text_y = self.button.factory.height
        gc.Scale(100 / float(self.zoom_x), 100 / float(self.zoom_y))

        if sys.platform == "darwin":
            gc.Translate(-x_offset + text_x, -y_offset + text_y)
            dc.DrawText(label_text, 0, 0)
        else:
            gc.Translate(-x_offset, -y_offset)
            dc.DrawText(label_text, text_x, text_y)

        if not self.button.enabled:
            self._draw_disable_mask(gc)

    def OnLeftDown(self, evt):
        # if the button is supposed to toggle, set the toggle_state
        # to the opposite of what it currently is
        if self.button.factory.toggle:
            self.toggle_state = not self.toggle_state

            if self.toggle_state:
                tooltip = wx.ToolTip(self.button.factory.toggle_tooltip)
            else:
                tooltip = wx.ToolTip(self.button.factory.tooltip)
            self.button.control.SetToolTip(tooltip)

        self.state = "down"
        evt.Skip()
        self.Refresh()

    def OnLeftUp(self, evt):
        self.state = "up"
        self.button.update_editor()
        evt.Skip()
        self.Refresh()

    def OnEnterWindow(self, evt):
        self.hover = True
        self.Refresh()

    def OnLeaveWindow(self, evt):
        self.hover = False
        self.Refresh()

    def OnWheel(self, evt):
        pass

    def _draw_disable_mask(self, gc):
        """ Draws a mask using the background color with the alpha
            set to about 33%
        """
        best_size = self.DoGetBestSize()

        path = gc.CreatePath()
        path.AddRectangle(0, 0, best_size.width, best_size.height)
        bgcolor = self.GetBackgroundColour()
        bgcolor.Set(bgcolor.red, bgcolor.green, bgcolor.blue, 175)
        gc.SetBrush(wx.Brush(bgcolor))
        gc.FillPath(path)

    def _draw_toggle(self, gc):
        # the toggle doc and button doc may not be the same
        # size, so calculate the scaling factor. Byt using the padding
        # to lie about the size of the toggle button, we can grow the
        # toggle a bit to use some of the padding. This is good for icons
        # which use all of their available space
        zoom_scale_x = float(self.zoom_x) / 100
        zoom_scale_y = float(self.zoom_y) / 100
        doc_size = self.document.getSize()
        toggle_doc_size = self.toggle_document.getSize()
        w_scale = (
            zoom_scale_x
            * doc_size[0]
            / (toggle_doc_size[0] - self.padding[0] - 1)
        )
        h_scale = (
            zoom_scale_y
            * doc_size[1]
            / (toggle_doc_size[1] - self.padding[1] - 1)
        )

        # move to the center of the allotted area
        best_size = self.DoGetBestSize()
        x_offset = (best_size.width - self.button.factory.width) / 2.0
        y_offset = self.padding[1] / 2.0
        gc.Translate(x_offset, y_offset)

        # Now scale the gc and render
        gc.Scale(w_scale, h_scale)
        self.toggle_document.render(gc)

        # And return the scaling factor back to what it originally was
        # and return to the origial location
        gc.Scale(1 / w_scale, 1 / h_scale)
        gc.Translate(-x_offset, -y_offset)


class SVGButtonEditor(Editor):
    """ Traits UI 'display only' image editor.
    """

    document = Instance(SVGDocument)
    toggle_document = Instance(SVGDocument)

    # -------------------------------------------------------------------------
    # Editor API
    # -------------------------------------------------------------------------

    def init(self, parent):
        """ Finishes initializing the editor by creating the underlying toolkit
            widget.
        """

        self.document = SVGDocument.createFromFile(
            self.factory.filename, renderer=Renderer
        )

        # load the button toggle document which will be displayed when the
        # button is toggled.
        if self.factory.toggle_filename:
            self.toggle_document = SVGDocument.createFromFile(
                self.factory.toggle_filename, renderer=Renderer
            )
        else:
            tree = etree.parse(
                os.path.join(
                    os.path.dirname(__file__), "data", "button_toggle.svg"
                )
            )
            self.toggle_document = SVGDocument(
                tree.getroot(), renderer=Renderer
            )

        padding = (self.factory.width_padding, self.factory.height_padding)
        self.control = ButtonRenderPanel(parent, self, padding=padding)

        if self.factory.tooltip != "":
            self.control.SetToolTip(wx.ToolTip(self.factory.tooltip))

        svg_w, svg_h = self.control.GetBestSize()
        self.control.zoom_x /= float(svg_w) / self.factory.width
        self.control.zoom_y /= float(svg_h) / self.factory.height
        self.control.Refresh()

    def prepare(self, parent):
        """ Finishes setting up the editor. This differs from the base class
            in that self.update_editor() is not called at the end, which
            would fire an event.
        """
        name = self.extended_name
        if name != "None":
            self.context_object.on_trait_change(
                self._update_editor, name, dispatch="ui"
            )
        self.init(parent)
        self._sync_values()

    def update_editor(self):
        """ Updates the editor when the object trait changes externally to the
            editor.
        """
        factory = self.factory
        self.value = factory.value
