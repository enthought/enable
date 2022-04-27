# (C) Copyright 2005-2022 Enthought, Inc., Austin, TX
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
        from pyface.ui.wx.font import weight_to_wx_weight
        import kiva.constants as kc

        font = editor.value
        weight = weight_to_wx_weight.get(
            font._get_weight(), wx.FONTWEIGHT_NORMAL)
        style = (
            wx.FONTSTYLE_ITALIC if font.style in kc.italic_styles
            else wx.FONTSTYLE_NORMAL
        )
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
        from pyface.ui.wx.font import wx_weight_to_weight
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
            weight=wx_weight_to_weight[font.GetWeight()],
            style=(
                # XXX: treat wx.FONTSTYLE_OBLIQUE as italic for now
                kc.NORMAL if font.GetStyle() == wx.FONTSTYLE_NORMAL
                else kc.ITALIC
            ),
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

        weight = " Bold" if font.is_bold() else ""
        style = " Italic" if font.style in kc.italic_styles else ""
        underline = " Underline" if font.underline else ""

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
        return sorted({f.fname for f in font_manager.ttf_db._entries})


def KivaFontEditor(*args, **traits):
    return ToolkitEditorFactory(*args, **traits)
