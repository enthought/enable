# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
import numpy as np

from kiva.fonttools.text._data import ENTRIES


class UnicodeAnalyzer:
    """ An object which, unlike `unicodedata`, will tell you the Unicode
    language group for the codepoints in a string.
    """
    def __init__(self):
        self.ranges = np.array([e[:2] for e in ENTRIES], dtype=np.int32)
        self.values = [e[2:] for e in ENTRIES]

    def languages(self, text):
        """ Given a Unicode string, return the languages that it contains.
        """
        result = []
        last_lang = ""
        last_start = 0

        # XXX: Should this be normalized first?
        for idx, cp in enumerate(text):
            lang, _ = self._lookup_codepoint(cp)
            if lang != last_lang:
                if idx > 0:
                    result.append((last_start, idx, last_lang))
                last_lang = lang
                last_start = idx

        result.append((last_start, idx + 1, last_lang))

        return result

    def _lookup_codepoint(self, cp):
        """ Look up a single codepoint in the database.
        """
        # `self.ranges` is an Nx2 numpy array of codepoint ranges. We subtract
        # the given codepoint to get an Nx2 array of offsets. The "bucket"
        # containing the given codepoint is the one whose start offset is zero
        # or negative and whose end is zero or positive. That should only be
        # True in one location, so we get the index of that location.
        comps = self.ranges - ord(cp)
        index = ((comps[:, 0] <= 0) == (comps[:, 1] >= 0)).argmax()
        return self.values[index]
