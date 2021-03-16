# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
import itertools
import os

from kiva.fonttools._constants import preferred_fonts


class FontEntry(object):
    """ A class for storing properties of fonts which have been discovered on
    the local machine.

    `__hash__` is implemented so that set() can be used to prune duplicates.
    """
    def __init__(self, fname="", family="", style="normal", variant="normal",
                 weight="normal", stretch="normal", size="medium",
                 face_index=0):

        self.fname = fname
        self.family = family
        self.style = style
        self.variant = variant
        self.weight = weight
        self.stretch = stretch
        self.face_index = face_index

        try:
            self.size = str(float(size))
        except ValueError:
            self.size = size

    def __hash__(self):
        c = tuple(getattr(self, k) for k in sorted(self.__dict__))
        return hash(c)

    def __repr__(self):
        fname = os.path.basename(self.fname)
        return (
            f"<FontEntry '{self.family}' ({fname}[{self.face_index}]) "
            f"{self.style} {self.variant} {self.weight} {self.stretch}>"
        )


class FontDatabase:
    """ A container for :class`FontEntry` instances of a specific type
    (TrueType/OpenType, AFM) which can be queried in different ways.
    """
    def __init__(self, entries):
        # Use a set to keep out the duplicates
        self._entries = {ent for ent in entries if isinstance(ent, FontEntry)}
        self._family_map = self._build_family_map(self._entries)
        self._file_map = self._build_file_map(self._entries)

    def add_fonts(self, entries):
        """ Add more :class`FontEntry` instances to the database.
        """
        for entry in entries:
            # Avoid non-FontEntry objects and duplicates
            if not isinstance(entry, FontEntry) or entry in self._entries:
                continue

            self._entries.add(entry)
            self._family_map.setdefault(entry.family, []).append(entry)
            self._file_map.setdefault(entry.fname, []).append(entry)

    def fonts_for_directory(self, directory):
        """ Returns all fonts whose file is in a directory.
        """
        result = []
        for fname, entries in self._file_map.items():
            if os.path.commonprefix([fname, directory]):
                result.extend(entries)
        return result

    def fonts_for_family(self, families):
        """ Returns all fonts which best match a particular family query or
        all possible fonts if exact families are not matched.

        `families` is a list of real and generic family names. An iterable
        of `FontEntry` instances is returned.
        """
        flat_list = (lambda it: list(itertools.chain.from_iterable(it)))

        # Translate generic families into lists of families
        fams = flat_list(preferred_fonts.get(fam, [fam]) for fam in families)
        # Then collect all entries for those families
        entries = flat_list(self._family_map.get(fam, []) for fam in fams)
        if entries:
            return entries

        # Return all entries if no families found
        # Yes, self._entries is a set. Consumers should only expect an iterable
        return self._entries

    def __len__(self):
        return len(self._entries)

    @staticmethod
    def _build_family_map(entries):
        ret = {}
        for entry in entries:
            ret.setdefault(entry.family, []).append(entry)

        return ret

    @staticmethod
    def _build_file_map(entries):
        ret = {}
        for entry in entries:
            ret.setdefault(entry.fname, []).append(entry)

        return ret
