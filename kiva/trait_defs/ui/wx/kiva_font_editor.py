# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
""" Defines the font editor factory for Kiva fonts, for the wxPython user
interface toolkit.
"""

import wx

from traits.trait_base import SequenceTypes
from traitsui.wx.font_editor import ToolkitEditorFactory as EditorFactory

from kiva.fonttools.font_manager import default_font_manager


# -----------------------------------------------------------------------------
#  'ToolkitEditorFactory' class:
# -----------------------------------------------------------------------------

class ToolkitEditorFactory(EditorFactory):
    """ wxPython editor factory for Kiva fonts.
    """

    # -------------------------------------------------------------------------
    #   Returns a Font's 'face name':
    # -------------------------------------------------------------------------

    def face_name(self, font):
        """ Returns a Font's typeface name.
        """
        face_name = font.face_name
        if isinstance(face_name, SequenceTypes):
            face_name = face_name[0]

        return face_name

    # -------------------------------------------------------------------------
    #  Returns a wxFont object corresponding to a specified object's font trait
    # -------------------------------------------------------------------------

    def to_wx_font(self, editor):
        """ Returns a wxFont object corresponding to a specified object's font
            trait.
        """
        import kiva.constants as kc

        font = editor.value
        weight = (
            wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD
        )[font.weight == kc.BOLD]
        style = (
            wx.FONTSTYLE_NORMAL, wx.FONTSTYLE_ITALIC
        )[font.style == kc.ITALIC]
        family = {
            kc.DEFAULT: wx.FONTFAMILY_DEFAULT,
            kc.DECORATIVE: wx.FONTFAMILY_DECORATIVE,
            kc.ROMAN: wx.FONTFAMILY_ROMAN,
            kc.SCRIPT: wx.FONTFAMILY_SCRIPT,
            kc.SWISS: wx.FONTFAMILY_SWISS,
            kc.MODERN: wx.FONTFAMILY_MODERN,
        }.get(font.family, wx.FONTFAMILY_SWISS)

        return wx.Font(
            font.size,
            family,
            style,
            weight,
            (font.underline != 0),
            self.face_name(font),
        )

    # -------------------------------------------------------------------------
    #  Gets the application equivalent of a wxPython value:
    # -------------------------------------------------------------------------

    def from_wx_font(self, font):
        """ Gets the application equivalent of a wxPython value.
        """
        import kiva.constants as kc
        from kiva.fonttools import Font

        return Font(
            size=font.GetPointSize(),
            family={
                wx.FONTFAMILY_DEFAULT: kc.DEFAULT,
                wx.FONTFAMILY_DECORATIVE: kc.DECORATIVE,
                wx.FONTFAMILY_ROMAN: kc.ROMAN,
                wx.FONTFAMILY_SCRIPT: kc.SCRIPT,
                wx.FONTFAMILY_SWISS: kc.SWISS,
                wx.FONTFAMILY_MODERN: kc.MODERN,
            }.get(font.GetFamily(), kc.SWISS),
            weight=(
                kc.NORMAL, kc.BOLD
            )[font.GetWeight() == wx.FONTWEIGHT_BOLD],
            style=(
                kc.NORMAL, kc.ITALIC
            )[font.GetStyle() == wx.FONTSTYLE_ITALIC],
            underline=font.GetUnderlined() - 0,  # convert Bool to an int type
            face_name=font.GetFaceName(),
        )

    # -------------------------------------------------------------------------
    #  Returns the text representation of the specified object trait value:
    # -------------------------------------------------------------------------

    def str_font(self, font):
        """ Returns the text representation of the specified object trait value
        """
        import kiva.constants as kc

        weight = {kc.BOLD: " Bold"}.get(font.weight, "")
        style = {kc.ITALIC: " Italic"}.get(font.style, "")
        underline = " Underline" if font.underline != 0 else ""

        return "%s point %s%s%s%s" % (
            font.size,
            self.face_name(font),
            style,
            weight,
            underline,
        )

    # -------------------------------------------------------------------------
    #  Returns a list of all available font facenames:
    # -------------------------------------------------------------------------

    def all_facenames(self):
        """ Returns a list of all available font typeface names.
        """
        font_manager = default_font_manager()
        return sorted({f.name for f in font_manager.ttflist})


def KivaFontEditor(*args, **traits):
    return ToolkitEditorFactory(*args, **traits)
