# (C) Copyright 2005-2022 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!

from pyface.font import Font as PyfaceFont
from pyface.font_dialog import get_font
from traits.api import Bool, Callable, Instance, Str, observe
from traits.trait_base import SequenceTypes
from traitsui.api import EditorFactory, Editor as BaseEditor, toolkit_object

from kiva.fonttools.font import Font
import kiva.constants as kc
from enable.component import Component
from enable.label import Label
from enable.tools.button_tool import ButtonTool
from enable.window import Window
from .editor_with_component import EditorWithLabelComponent


def face_name(font):
    """ Returns a Font's typeface name.
    """
    face_name = font.face_name
    if isinstance(face_name, SequenceTypes):
        face_name = face_name[0]

    return face_name


def str_font(font):
    """ Returns the text representation of the specified font trait value
    """

    weight = " Bold" if font.is_bold() else ""
    style = " Italic" if font.style in kc.italic_styles else ""
    underline = " Underline" if font.underline else ""

    return f"{font.size} point {face_name(font)}{weight}{style}{underline}".strip()


class ReadOnlyEditor(EditorWithLabelComponent):
    """An Editor which displays a label using the font."""

    def init(self, parent):
        self.font = self.value
        super().init(parent)

    def update_editor(self):
        self.font = self.value
        super().update_editor()

    def string_value(self, value, format_func=None):
        if self.factory.sample_text:
            return self.factory.sample_text

        return super().string_value(value, str_font)


class SimpleEditor(ReadOnlyEditor):
    """An Editor which displays a label using the font, click for font dialog.
    """

    button = Instance(ButtonTool)

    def create_component(self):
        component = super().create_component()
        # add a grey border to indicate interactivity
        component.border_visible = True
        component.border_width = 1
        component.border_color = (0.5, 0.5, 0.5, 1.0)

        # add a button tool to make the label respond to clicks
        self.button = ButtonTool(component=component)
        component.tools.append(self.button)
        return component

    def update_object(self, value):
        """Handle changes to the font due to user action.
        """
        self.value = value
        # force a refresh of the component's settings
        self.update_editor()

    @observe('button:clicked')
    def button_clicked(self, event):
        if self.window is None:
            return
        pyface_font = PyfaceFont(
            family=[self.value.face_name],
            weight=str(self.value.weight),
            style='italic' if self.value.style in kc.italic_styles else 'normal',
            size=self.value.size,
        )
        pyface_font = get_font(self.window.control, pyface_font)
        if pyface_font is not None:
            font = Font(
                face_name=pyface_font.family[0],
                weight=pyface_font.weight_,
                style=kc.ITALIC if pyface_font.style == 'italic' else kc.NORMAL,
                size=int(pyface_font.size),
            )
            self.update_object(font)


class KivaFontEditor(EditorFactory):
    """Editor factory for KivaFontEditors
    """

    #: Alternative text to display instead of the font description.
    sample_text = Str()

    #: Switch to turn off high resolution rendering if needed.
    high_resolution = Bool(True)

    #: The default format func displays a description of the font.
    format_func = Callable(str_font)

    def _get_simple_editor_class(self):
        return SimpleEditor

    def _get_readonly_editor_class(self):
        return ReadOnlyEditor
