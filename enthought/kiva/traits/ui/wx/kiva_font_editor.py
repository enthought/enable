#------------------------------------------------------------------------------
# Copyright (c) 2005-2007 by Enthought, Inc.
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in enthought/LICENSE.txt and may be redistributed only
# under the conditions described in the aforementioned license.  The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
# Thanks for using Enthought open source!
#
#------------------------------------------------------------------------------

""" Defines the font editor factory for Kiva fonts, for the wxPython user
interface toolkit.
"""


#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

import wx

from enthought.traits.trait_base \
    import SequenceTypes

from enthought.traits.ui.editors.font_editor \
    import ToolkitEditorFactory as EditorFactory

from enthought.traits.ui.wx.font_editor \
    import SimpleFontEditor, CustomFontEditor, TextFontEditor, \
    ReadonlyFontEditor
 
from enthought.kiva.fonttools.font_manager import fontManager


#-------------------------------------------------------------------------------
#  'ToolkitEditorFactory' class:
#-------------------------------------------------------------------------------

class ToolkitEditorFactory ( EditorFactory ):
    """ wxPython editor factory for Kiva fonts.
    """
    
    def _get_simple_editor_class(self):
        return SimpleEditor
    
    def _get_custom_editor_class(self):
        return CustomEditor
    
    def _get_text_editor_class(self):
        return TextEditor
    
    def _get_readonly_editor_class(self):
        return ReadonlyEditor

# Common functions used by the Editor objects.    
#---------------------------------------------------------------------------
#   Returns a Font's 'face name':
#---------------------------------------------------------------------------

def face_name ( font ):
    """ Returns a Font's typeface name.
    """
    face_name = font.face_name
    if type( face_name ) in SequenceTypes:
        face_name = face_name[0]

    return face_name

#---------------------------------------------------------------------------
#  Returns a wxFont object corresponding to a specified object's font trait:
#---------------------------------------------------------------------------

def to_wx_font ( editor ):
    """ Returns a wxFont object corresponding to a specified object's font
        trait.
    """
    import enthought.kiva.constants as kc

    font   = editor.value
    weight = ( wx.NORMAL, wx.BOLD   )[ font.weight == kc.BOLD ]
    style  = ( wx.NORMAL, wx.ITALIC )[ font.style  == kc.ITALIC ]
    family = { kc.DEFAULT:    wx.DEFAULT,
               kc.DECORATIVE: wx.DECORATIVE,
               kc.ROMAN:      wx.ROMAN,
               kc.SCRIPT:     wx.SCRIPT,
               kc.SWISS:      wx.SWISS,
               kc.MODERN:     wx.MODERN }.get( font.family, wx.SWISS )

    return wx.Font( font.size, family, style, weight,
                    (font.underline != 0), face_name( font ) )

#---------------------------------------------------------------------------
#  Gets the application equivalent of a wxPython value:
#---------------------------------------------------------------------------

def from_wx_font ( font ):
    """ Gets the application equivalent of a wxPython value.
    """
    import enthought.kiva.constants as kc
    from enthought.kiva.fonttools import Font

    return Font( size = font.GetPointSize(),
                 family = { wx.DEFAULT:    kc.DEFAULT,
                            wx.DECORATIVE: kc.DECORATIVE,
                            wx.ROMAN:      kc.ROMAN,
                            wx.SCRIPT:     kc.SCRIPT,
                            wx.SWISS:      kc.SWISS,
                            wx.MODERN:     kc.MODERN }.get( font.GetFamily(),
                                                            kc.SWISS ),
                 weight = ( kc.NORMAL, kc.BOLD   )[ font.GetWeight() == wx.BOLD ],
                 style = ( kc.NORMAL, kc.ITALIC )[ font.GetStyle()  == wx.ITALIC ],
                 underline = font.GetUnderlined() - 0, #convert Bool to an int type
                 face_name = font.GetFaceName() )

#---------------------------------------------------------------------------
#  Returns the text representation of the specified object trait value:
#---------------------------------------------------------------------------

def str_font ( font ):
    """ Returns the text representation of the specified object trait value.
    """
    import enthought.kiva.constants as kc

    weight    = { kc.BOLD:   ' Bold'   }.get( font.weight, '' )
    style     = { kc.ITALIC: ' Italic' }.get( font.style,  '' )
    underline = [ '', ' Underline' ][ font.underline != 0 ]

    return '%s point %s%s%s%s' % (
           font.size, face_name( font ), style, weight, underline )

#---------------------------------------------------------------------------
#  Returns a list of all available font facenames:
#---------------------------------------------------------------------------

def all_facenames ( ):
    """ Returns a list of all available font typeface names.
    """
    facenames = fontManager.ttfdict.keys()
    return facenames
    
# Define the Editor classes.    
class SimpleEditor(SimpleFontEditor):
    # This class has been defined here simply so its uses the functions
    # 'to_wx_font', etc. declared in this file instead of the ones in
    # enthought.traits.ui.wx.font_editor.
    pass

class CustomEditor(CustomFontEditor):
    # This class has been defined here simply so its uses the functions
    # 'to_wx_font', etc. declared in this file instead of the ones in
    # enthought.traits.ui.wx.font_editor.
    pass

class TextEditor(TextFontEditor):
    # This class has been defined here simply so its uses the functions
    # 'to_wx_font', etc. declared in this file instead of the ones in
    # enthought.traits.ui.wx.font_editor.
    pass


class ReadonlyEditor(ReadonlyFontEditor):
    # This class has been defined here simply so its uses the functions
    # 'to_wx_font', etc. declared in this file instead of the ones in
    # enthought.traits.ui.wx.font_editor.
    pass

KivaFontEditor = ToolkitEditorFactory

