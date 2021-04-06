# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
import locale

from kiva.fonttools.text._data import SCRIPTS

# Derived from kiva.fonttools._util:
# `_ot_code_page_masks` and `_ot_unicode_range_bits`
# These are the font languages which we recognize
_FONT_LANGUAGES = [
    "Arabic", "Armenian", "Balinese", "Bengali", "Buginese",
    "Canadian_Aboriginal", "Cherokee", "Coptic", "Cyrillic", "Deseret",
    "Devanagari", "Ethiopic", "Georgia", "Glagolitic", "Gothic", "Greek",
    "Gujarati", "Gurmukhi", "Hebrew", "Japanese", "Kannada", "Khmer", "Korean",
    "Lao", "Latin", "Limbu", "Malayalam", "Math", "Mongolian", "Myanmar",
    "New_Tai_Lue", "Nko", "Ogham", "Oriya", "Phoenician", "Runic",
    "Simplified Chinese", "Sinhala", "Symbol", "Syriac", "Tai_Le", "Tamil",
    "Telugu", "Thaana", "Thai", "Tibetan", "Tifinagh", "Traditional Chinese",
    "Vai", "Vietnamese",
]


def build_script_to_language_map():
    """ Create a dictionary which maps from script name (from `SCRIPTS`) to
    font language.

    NOTE: The langauge for a given script is locale dependent.
    """
    locale_lang = locale.getdefaultlocale()[0]

    if locale_lang == "C":
        locale_lang = "en_US"

    # Pick a language to use for "Han" script
    han_lang = "Traditional Chinese"  # Default
    if locale_lang in ("zh_CN", "zh_SG"):
        han_lang = "Simplified Chinese"
    elif locale_lang.startswith("ja"):
        han_lang = "Japanese"
    elif locale_lang.startswith("ko"):
        han_lang = "Korean"

    # Mapping from script -> langauge that we're _mostly_ sure about
    known_mappings = {
        # Special script properties
        "Common": "Common",
        "Inherited": "Inherited",
        "Unknown": "Unknown",

        # Scripts which infer the writing system
        "Bopomofo": "Traditional Chinese",  # XXX: Taiwan only?
        "Han": han_lang,
        "Hangul": "Korean",
        "Hiragana": "Japanese",
        "Katakana": "Japanese",
    }

    mapping = {}
    for script in SCRIPTS:
        if script in known_mappings:
            mapping[script] = known_mappings[script]
        elif script in _FONT_LANGUAGES:
            mapping[script] = script
        else:
            mapping[script] = "Latin"

    return mapping
