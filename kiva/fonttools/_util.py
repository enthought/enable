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


def get_ttf_prop_dict(font):
    """ Parse the 'name' table of a :class:`TTFont` instance.
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
