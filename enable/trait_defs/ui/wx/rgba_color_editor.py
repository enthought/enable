# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
""" Defines the various RGBA color editors and the color editor factory, for
the wxPython user interface toolkit.
"""

import wx

from enable.colors import color_table
from enable.label import Label
from enable.window import Window
from traits.api import Bool
from traits.trait_base import SequenceTypes

from traitsui.api import EditorFactory, View
from traitsui.wx.editor import Editor
from traitsui.wx.editor_factory import ReadonlyEditor
from traitsui.wx.helper import position_window

# -----------------------------------------------------------------------------
#  Constants:
# -----------------------------------------------------------------------------

# Standard color samples:
COLOR_CHOICES = (0, 51, 102, 153, 204, 255)
COLOR_SAMPLES = tuple(
    [
        wx.Colour(r, g, b)
        for r in COLOR_CHOICES
        for g in COLOR_CHOICES
        for b in COLOR_CHOICES
    ]
)

# -----------------------------------------------------------------------------
#  'ToolkitEditorFactory' class:
# -----------------------------------------------------------------------------


class ToolkitEditorFactory(EditorFactory):
    """ wxPython editor factory for RGBA color editors.
    """

    # -------------------------------------------------------------------------
    #  Trait definitions:
    # -------------------------------------------------------------------------

    # Is the underlying color trait mapped?
    mapped = Bool(True)

    # -------------------------------------------------------------------------
    #  Traits view definition:
    # -------------------------------------------------------------------------

    traits_view = View(["mapped{Is the value mapped?}", "|[]>"])

    # -------------------------------------------------------------------------
    #  'Editor' factory methods:
    # -------------------------------------------------------------------------

    def simple_editor(self, ui, object, name, description, parent):
        return SimpleColorEditor(
            parent,
            factory=self,
            ui=ui,
            object=object,
            name=name,
            description=description,
        )

    def custom_editor(self, ui, object, name, description, parent):
        return CustomColorEditor(
            parent,
            factory=self,
            ui=ui,
            object=object,
            name=name,
            description=description,
        )

    def text_editor(self, ui, object, name, description, parent):
        return TextColorEditor(
            parent,
            factory=self,
            ui=ui,
            object=object,
            name=name,
            description=description,
        )

    def readonly_editor(self, ui, object, name, description, parent):
        return ReadonlyColorEditor(
            parent,
            factory=self,
            ui=ui,
            object=object,
            name=name,
            description=description,
        )

    # -------------------------------------------------------------------------
    #  Gets the object trait color:
    # -------------------------------------------------------------------------

    def get_color(self, editor):
        """ Gets the object trait color.
        """
        if self.mapped:
            return getattr(editor.object, editor.name + "_")
        return getattr(editor.object, editor.name)

    # -------------------------------------------------------------------------
    #  Gets the wxPython color equivalent of the object trait:
    # -------------------------------------------------------------------------

    def to_wx_color(self, editor):
        """ Gets the wxPython color equivalent of the object trait.
        """
        r, g, b, a = self.get_color(editor)
        return wx.Colour(int(r * 255.0), int(g * 255.0), int(b * 255.0))

    # -------------------------------------------------------------------------
    #  Gets the application equivalent of a wxPython value:
    # -------------------------------------------------------------------------

    def from_wx_color(self, color):
        """ Gets the application equivalent of a wxPython color value.
        """
        return (
            color.Red() / 255.0,
            color.Green() / 255.0,
            color.Blue() / 255.0,
            1.0,
        )

    # -------------------------------------------------------------------------
    #  Returns the text representation of a specified color value:
    # -------------------------------------------------------------------------

    def str_color(self, color):
        """ Returns the text representation of a specified color value.
        """
        if type(color) in SequenceTypes:
            return "(%d,%d,%d,%d)" % (
                int(color[0] * 255.0),
                int(color[1] * 255.0),
                int(color[2] * 255.0),
                int(color[3] * 255.0),
            )
        return color


# -----------------------------------------------------------------------------
#  'SimpleColorEditor' class:
# -----------------------------------------------------------------------------


class SimpleColorEditor(Editor):
    """ Simple style of editor for RGBA colors, which displays a text field
    containing the string representation of the color value, and whose
    background color is the selected color. Clicking in the text
    field opens a dialog box to select a color value.
    """

    # -------------------------------------------------------------------------
    #  Finishes initializing the editor by creating the underlying toolkit
    #  widget:
    # -------------------------------------------------------------------------

    def init(self, parent):
        """ Finishes initializing the editor by creating the underlying toolkit
            widget.
        """
        window = Window(
            parent, component=Label("", border_size=1, font="modern 9")
        )
        self._swatch = window.component
        self.control = window.control
        self.control.SetSize((110, 20))
        window.component.observe(
            self.popup_editor, "left_up", dispatch="ui"
        )

    # -------------------------------------------------------------------------
    #  Invokes the pop-up editor for an object trait:
    # -------------------------------------------------------------------------

    def popup_editor(self, event):
        """ Invokes the pop-up editor for an object trait.
        """
        if not hasattr(self.control, "is_custom"):
            self._popup_dialog = ColorDialog(self)
        else:
            update_handler = self.control.update_handler
            if update_handler is not None:
                update_handler(False)
            color_data = wx.ColourData()
            color_data.SetColour(self.factory.to_wx_color(self))
            color_data.SetChooseFull(True)
            dialog = wx.ColourDialog(self.control, color_data)
            position_window(dialog, parent=self.control)
            if dialog.ShowModal() == wx.ID_OK:
                self.value = self.factory.from_wx_color(
                    dialog.GetColourData().GetColour()
                )
                self.update_editor()
            dialog.Destroy()
            if update_handler is not None:
                update_handler(True)

    # -------------------------------------------------------------------------
    #  Updates the object trait when a color swatch is clicked:
    # -------------------------------------------------------------------------

    def update_object_from_swatch(self, event):
        """ Updates the object trait when a color swatch is clicked.
        """
        control = event.GetEventObject()
        r, g, b, a = self.factory.from_wx_color(control.GetBackgroundColour())
        self.value = (r, g, b, self.factory.get_color(self)[3])
        self.update_editor()

    # -------------------------------------------------------------------------
    #  Updates the object trait when the alpha channel slider is scrolled:
    # -------------------------------------------------------------------------

    def update_object_from_scroll(self, event):
        """ Updates the object trait when the alpha channel slider is scrolled.
        """
        control = event.GetEventObject()
        r, g, b, a = self.factory.get_color(self)
        self.value = (r, g, b, (100 - control.GetValue()) / 100.0)
        self.update_editor()

    # -------------------------------------------------------------------------
    #  Updates the editor when the object trait changes external to the editor:
    # -------------------------------------------------------------------------

    def update_editor(self):
        """ Updates the editor when the object trait changes externally to the
            editor.
        """
        alpha = self.factory.get_color(self)[3]
        self._swatch.text = self.str_value
        if self._slider is not None:
            self._slider.SetValue(100 - int(alpha * 100.0))
        set_color(self)

    # -------------------------------------------------------------------------
    #  Returns the text representation of a specified color value:
    # -------------------------------------------------------------------------

    def string_value(self, color):
        """ Returns the text representation of a specified color value.
        """
        return self.factory.str_color(color)


# -----------------------------------------------------------------------------
#  'CustomColorEditor' class:
# -----------------------------------------------------------------------------


class CustomColorEditor(SimpleColorEditor):
    """ Custom style of editor for RGBA colors, which displays a large color
    swatch with the selected color, a set of color chips of other possible
    colors, and a vertical slider for the alpha value.
    """

    # -------------------------------------------------------------------------
    #  Finishes initializing the editor by creating the underlying toolkit
    #  widget:
    # -------------------------------------------------------------------------

    def init(self, parent):
        """ Finishes initializing the editor by creating the underlying toolkit
            widget.
        """
        self.control = color_editor_for(self, parent)

    # -------------------------------------------------------------------------
    #  Disposes of the contents of an editor:
    # -------------------------------------------------------------------------

    def dispose(self):
        """ Disposes of the contents of an editor.
        """
        self.control._swatch_editor.dispose()
        super().dispose()

    # -------------------------------------------------------------------------
    #  Updates the editor when the object trait changes external to the editor:
    # -------------------------------------------------------------------------

    def update_editor(self):
        """ Updates the editor when the object trait changes externally to the
            editor.
        """
        pass


# -----------------------------------------------------------------------------
#  'TextColorEditor' class:
# -----------------------------------------------------------------------------


class TextColorEditor(SimpleColorEditor):
    """ Text style of RGBA color editor, identical to the simple style.
    """

    pass


# -----------------------------------------------------------------------------
#  'ReadonlyColorEditor' class:
# -----------------------------------------------------------------------------


class ReadonlyColorEditor(ReadonlyEditor):
    """ Read-only style of RGBA color editor, which displays a read-only text
    field containing the string representation of the color value, and whose
    background color is the selected color.
    """

    # -------------------------------------------------------------------------
    #  Finishes initializing the editor by creating the underlying toolkit
    #  widget:
    # -------------------------------------------------------------------------

    def init(self, parent):
        """ Finishes initializing the editor by creating the underlying toolkit
            widget.
        """
        window = Window(
            parent, component=Label("", border_size=1, font="modern 9")
        )
        self._swatch = window.component
        self.control = window.control
        self.control.SetSize((110, 20))

    # -------------------------------------------------------------------------
    #  Returns the text representation of a specified color value:
    # -------------------------------------------------------------------------

    def string_value(self, color):
        """ Returns the text representation of a specified color value.
        """
        return self.factory.str_color(color)

    # -------------------------------------------------------------------------
    #  Updates the editor when the object trait changes external to the editor:
    # -------------------------------------------------------------------------

    def update_editor(self):
        """ Updates the editor when the object trait changes externally to the
            editor.
        """
        self._swatch.text = self.str_value
        set_color(self)


# -----------------------------------------------------------------------------
#   Sets the color of the specified editor's color control:
# -----------------------------------------------------------------------------


def set_color(editor):
    """  Sets the color of the specified color control.
    """
    color = editor.factory.get_color(editor)
    control = editor._swatch
    control.bg_color = color
    if (color[0] > 0.75) or (color[1] > 0.75) or (color[2] > 0.75):
        control.color = color_table["black"]
    else:
        control.color = color_table["white"]


# ----------------------------------------------------------------------------
#  Creates a custom color editor panel for a specified editor:
# ----------------------------------------------------------------------------


def color_editor_for(editor, parent, update_handler=None):
    """ Creates a custom color editor panel for a specified editor.
    """
    # Create a panel to hold all of the buttons:
    panel = wx.Panel(parent, -1)
    sizer = wx.BoxSizer(wx.HORIZONTAL)
    panel._swatch_editor = swatch_editor = editor.factory.simple_editor(
        editor.ui, editor.object, editor.name, editor.description, panel
    )
    swatch_editor.prepare(panel)
    control = swatch_editor.control
    control.is_custom = True
    control.update_handler = update_handler
    control.SetSize(wx.Size(110, 72))
    sizer.Add(control, 1, wx.EXPAND | wx.RIGHT, 4)

    # Add all of the color choice buttons:
    sizer2 = wx.GridSizer(0, 12, 0, 0)

    for color_sample in COLOR_SAMPLES:
        control = wx.Button(panel, -1, "", size=wx.Size(18, 18))
        control.SetBackgroundColour(color_sample)
        control.update_handler = update_handler
        panel.Bind(
            wx.EVT_BUTTON,
            swatch_editor.update_object_from_swatch,
            id=control.GetId(),
        )
        sizer2.Add(control)
        editor.set_tooltip(control)

    sizer.Add(sizer2)

    alpha = editor.factory.get_color(editor)[3]
    swatch_editor._slider = slider = wx.Slider(
        panel,
        -1,
        100 - int(alpha * 100.0),
        0,
        100,
        size=wx.Size(20, 40),
        style=wx.SL_VERTICAL | wx.SL_AUTOTICKS,
    )
    slider.SetTickFreq(10)
    slider.SetPageSize(10)
    slider.SetLineSize(1)
    slider.Bind(wx.EVT_SCROLL, swatch_editor.update_object_from_scroll)
    sizer.Add(slider, 0, wx.EXPAND | wx.LEFT, 6)

    # Set-up the layout:
    panel.SetSizerAndFit(sizer)

    # Return the panel as the result:
    return panel


# -----------------------------------------------------------------------------
#  'ColorDialog' class:
# -----------------------------------------------------------------------------


class ColorDialog(wx.Frame):
    """ Dialog box for selecting a color value.
    """

    # -------------------------------------------------------------------------
    #  Initializes the object:
    # -------------------------------------------------------------------------

    def __init__(self, editor):
        """ Initializes the object.
        """
        wx.Frame.__init__(self, editor.control, -1, "", style=wx.SIMPLE_BORDER)
        wx.EVT_ACTIVATE(self, self._on_close_dialog)
        self._closed = False
        self._closeable = True

        panel = color_editor_for(editor, self, self._close_dialog)
        self._swatch_editor = panel._swatch_editor

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(panel)
        self.SetSizerAndFit(sizer)
        position_window(self, parent=editor.control)
        self.Show()

    # -------------------------------------------------------------------------
    #  Closes the dialog:
    # -------------------------------------------------------------------------

    def _on_close_dialog(self, event, rc=False):
        """ Closes the dialog.
        """
        if not event.GetActive():
            self._close_dialog()

    # -------------------------------------------------------------------------
    #  Closes the dialog:
    # -------------------------------------------------------------------------

    def _close_dialog(self, closeable=None):
        """ Closes the dialog.
        """
        if closeable is not None:
            self._closeable = closeable
        if self._closeable and (not self._closed):
            self._closed = True
            self._swatch_editor.dispose()
            self.Destroy()


def RGBAColorEditor(*args, **traits):
    return ToolkitEditorFactory(*args, **traits)
