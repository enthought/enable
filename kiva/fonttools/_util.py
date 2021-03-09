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


def get_ttf_prop_dict(font):
    """ Return the property dictionary from a :class:`TTFont` instance.
    """
    n = font["name"]
    propdict = {}
    for prop in n.names:
        try:
            if "name" in propdict and "sfnt4" in propdict:
                break
            elif prop.nameID == 1 and "name" not in propdict:
                propdict["name"] = _decode_prop(prop.string)
            elif prop.nameID == 4 and "sfnt4" not in propdict:
                propdict["sfnt4"] = _decode_prop(prop.string)
        except UnicodeDecodeError:
            continue

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


def _decode_prop(prop):
    """ Decode a prop string.

    Parameters
    ----------
    prop : bytestring

    Returns
    -------
    string
    """
    # Adapted from: https://gist.github.com/pklaus/dce37521579513c574d0
    encoding = "utf-16-be" if b"\x00" in prop else "utf-8"
    return prop.decode(encoding)
