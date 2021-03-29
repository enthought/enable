# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
from kiva.fonttools._constants import weight_dict

# Unicode & Apple
_plat_ids = (0, 1)
_english_id = 0
# MS
# https://docs.microsoft.com/en-us/typography/opentype/spec/name#windows-language-ids  # noqa: E501
_ms_plat_id = 3
_ms_english_ids = {
    0x0C09: "Australia",
    0x2809: "Belize",
    0x1009: "Canada",
    0x2409: "Caribbean",
    0x4009: "India",
    0x1809: "Ireland",
    0x2009: "Jamaica",
    0x4409: "Malaysia",
    0x1409: "New Zealand",
    0x3409: "Republic of the Philippines",
    0x4809: "Singapore",
    0x1C09: "South Africa",
    0x2C09: "Trinidad and Tobago",
    0x0809: "United Kingdom",
    0x0409: "United States",
    0x3009: "Zimbabwe",
}
# TrueType 'name' table IDs
# https://developer.apple.com/fonts/TrueType-Reference-Manual/RM06/Chap6name.html  # noqa: E501
_name_ids = {
    0: "copyright",
    1: "family",
    2: "style",
    3: "unique_subfamily_id",
    4: "full_name",
    5: "version",
    6: "postscript_name",
}
# OpenType 'OS/2' table data
# https://docs.microsoft.com/en-us/typography/opentype/spec/os2
# `ulCodePageRange` bit meanings
_ot_code_page_masks = {
    "Latin": 0x93,
    "Cyrillic": 0x4,
    "Greek": 0x8,
    "Hebrew": 0x20,
    "Arabic": 0x40,
    "Vietnamese": 0x100,
    "Thai": 0x10000,
    "Japanese": 0x20000,
    "Simplified Chinese": 0x40000,
    "Traditional Chinese": 0x100000,
    "Korean": 0x280000,
    "Symbol": 0x80000000,
}
# `ulUnicodeRange` bit meanings
_ot_unicode_range_bits = {
    0: "Latin",
    1: "Latin",
    2: "Latin",
    3: "Latin",
    7: "Greek",
    8: "Coptic",
    9: "Cyrillic",
    10: "Armenian",
    11: "Hebrew",
    12: "Vai",
    13: "Arabic",
    14: "Nko",
    15: "Devanagari",
    16: "Bengali",
    17: "Gurmukhi",
    18: "Gujarati",
    19: "Oriya",
    20: "Tamil",
    21: "Telugu",
    22: "Kannada",
    23: "Malayalam",
    24: "Thai",
    25: "Lao",
    26: "Georgia",
    27: "Balinese",
    28: "Korean",
    29: "Latin",
    30: "Greek",
    38: "Math",
    52: "Korean",
    56: "Korean",
    58: "Phoenician",
    70: "Tibetan",
    71: "Syriac",
    72: "Thaana",
    73: "Sinhala",
    74: "Myanmar",
    75: "Ethiopic",
    76: "Cherokee",
    77: "Canadian_Aboriginal",
    78: "Ogham",
    79: "Runic",
    80: "Khmer",
    81: "Mongolian",
    86: "Gothic",
    87: "Deseret",
    93: "Limbu",
    94: "Tai_Le",
    95: "New_Tai_Lue",
    96: "Buginese",
    97: "Glagolitic",
    98: "Tifinagh",
}


def get_ttf_prop_dict(font):
    """ Parse the 'name' and 'OS/2' tables of a :class:`TTFont` instance.
    """
    propdict = {}
    table = font["name"]
    for rec in table.names:
        # We only care about records in English
        plat, lang = rec.platformID, rec.langID
        if not ((plat in _plat_ids and lang == _english_id)
                or (plat == _ms_plat_id and lang in _ms_english_ids)):
            continue
        # And of those, just the ones we have names for
        if rec.nameID not in _name_ids:
            continue

        # Convert the nameID to a nice string
        key = _name_ids[rec.nameID]
        # Skip duplicate records
        if key in propdict:
            continue

        # Use the NameRecord's toStr() method instead of ad-hoc decoding
        propdict[key] = rec.toStr()

    # NOTE: Not all fonts have the "OS/2" table, but it's the easiest to
    # extract language support information from.
    propdict["languages"] = set()
    try:
        table = font["OS/2"]
        languages = set()
        # Check the unicode range bits
        unicode_bits = table.getUnicodeRanges()
        if unicode_bits:
            for bit in unicode_bits:
                if bit in _ot_unicode_range_bits:
                    languages.add(_ot_unicode_range_bits[bit])
        # Check the codepage range bits
        cp_bits = table.ulCodePageRange1
        for lang, mask in _ot_code_page_masks.items():
            if cp_bits & mask:
                languages.add(lang)
        # Lock the set so that it's hashable
        propdict["languages"] = frozenset(languages)
    except KeyError:
        pass

    # Make sure "languages" is never empty
    if not propdict["languages"]:
        propdict["languages"] = frozenset(["Unknown"])

    return propdict


def weight_as_number(weight):
    """ Return the weight property as a numeric value.

    String values are converted to their corresponding numeric value.
    """
    allowed_weights = set(weight_dict.values())
    if isinstance(weight, str):
        try:
            weight = weight_dict[weight.lower()]
        except KeyError:
            weight = weight_dict["regular"]
    elif weight in allowed_weights:
        pass
    else:
        raise ValueError("weight not a valid integer")
    return weight
