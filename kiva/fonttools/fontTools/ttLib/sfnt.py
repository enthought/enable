"""ttLib/sfnt.py -- low-level module to deal with the sfnt file format.

Defines two public classes:
        SFNTReader
        SFNTWriter

(Normally you don't have to use these classes explicitly; they are
used automatically by ttLib.TTFont.)

The reading and writing of sfnt files is separated in two distinct
classes, since whenever to number of tables changes or whenever
a table's length chages you need to rewrite the whole file anyway.
"""

import struct
from kiva.fonttools import sstruct
import numpy
import os

# magic value corresponding to 0xb1b0afba; used in lieu of the hex
# constant to avoid signedness issues with python 2.4
CHECKSUM_MAGIC = -1313820742


class SFNTReader:

        def __init__(self, file, checkChecksums=1):
                self.file = file
                self.checkChecksums = checkChecksums
                data = self.file.read(sfntDirectorySize)
                if len(data) != sfntDirectorySize:
                        from kiva.fonttools.fontTools import ttLib
                        raise ttLib.TTLibError("Not a TrueType or OpenType font (not enough data)")
                sstruct.unpack(sfntDirectoryFormat, data, self)
                if self.sfntVersion not in ("\000\001\000\000", "OTTO", "true"):
                        from kiva.fonttools.fontTools import ttLib
                        raise ttLib.TTLibError("Not a TrueType or OpenType font (bad sfntVersion)")
                self.tables = {}
                for i in range(self.numTables):
                        entry = SFNTDirectoryEntry()
                        entry.fromFile(self.file)
                        if entry.length > 0:
                                self.tables[entry.tag] = entry
                        else:
                                # Ignore zero-length tables. This doesn't seem to be documented,
                                # yet it's apparently how the Windows TT rasterizer behaves.
                                # Besides, at least one font has been sighted which actually
                                # *has* a zero-length table.
                                pass

        def __contains__(self, item):
                return item in self.tables

        def keys(self):
                return self.tables.keys()

        def __getitem__(self, tag):
                """Fetch the raw table data."""
                entry = self.tables[tag]
                self.file.seek(entry.offset)
                data = self.file.read(entry.length)
                if self.checkChecksums:
                        if tag == 'head':
                                # Beh: we have to special-case the 'head' table.
                                checksum = calcChecksum(data[:8] + '\0\0\0\0' + data[12:])
                        else:
                                checksum = calcChecksum(data)
                        if self.checkChecksums > 1:
                                # Be obnoxious, and barf when it's wrong
                                assert checksum == entry.checksum, "bad checksum for '%s' table" % tag
                        elif checksum != entry.checkSum:
                                # Be friendly, and just print a warning.
                                print("bad checksum for '%s' table" % tag)
                return data

        def __delitem__(self, tag):
                del self.tables[tag]

        def close(self):
                self.file.close()


class SFNTWriter:

        def __init__(self, file, numTables, sfntVersion="\000\001\000\000"):
                self.file = file
                self.numTables = numTables
                self.sfntVersion = sfntVersion
                self.searchRange, self.entrySelector, self.rangeShift = getSearchRange(numTables)
                self.nextTableOffset = sfntDirectorySize + numTables * sfntDirectoryEntrySize
                # clear out directory area
                self.file.seek(self.nextTableOffset)
                # make sure we're actually where we want to be. (XXX old cStringIO bug)
                self.file.write('\0' * (self.nextTableOffset - self.file.tell()))
                self.tables = {}

        def __setitem__(self, tag, data):
                """Write raw table data to disk."""
                if tag in self.tables:
                        # We've written this table to file before. If the length
                        # of the data is still the same, we allow overwriting it.
                        entry = self.tables[tag]
                        if len(data) != entry.length:
                                from kiva.fonttools.fontTools import ttLib
                                raise ttLib.TTLibError("cannot rewrite '%s' table: length does not match directory entry" % tag)
                else:
                        entry = SFNTDirectoryEntry()
                        entry.tag = tag
                        entry.offset = self.nextTableOffset
                        entry.length = len(data)
                        self.nextTableOffset = self.nextTableOffset + ((len(data) + 3) & ~3)
                self.file.seek(entry.offset)
                self.file.write(data)
                self.file.seek(self.nextTableOffset)
                # make sure we're actually where we want to be. (XXX old cStringIO bug)
                self.file.write('\0' * (self.nextTableOffset - self.file.tell()))

                if tag == 'head':
                        entry.checkSum = calcChecksum(data[:8] + '\0\0\0\0' + data[12:])
                else:
                        entry.checkSum = calcChecksum(data)
                self.tables[tag] = entry

        def close(self, closeStream=1):
                """All tables must have been written to disk. Now write the
                directory.
                """
                tables = self.tables.items()
                tables.sort()
                if len(tables) != self.numTables:
                        from kiva.fonttools.fontTools import ttLib
                        raise ttLib.TTLibError("wrong number of tables; expected %d, found %d" % (self.numTables, len(tables)))

                directory = sstruct.pack(sfntDirectoryFormat, self)

                self.file.seek(sfntDirectorySize)
                for tag, entry in tables:
                        directory = directory + entry.toString()
                self.calcMasterChecksum(directory)
                self.file.seek(0)
                self.file.write(directory)
                if closeStream:
                        self.file.close()

        def calcMasterChecksum(self, directory):
                # calculate checkSumAdjustment
                tags = self.tables.keys()
                checksums = numpy.zeros(len(tags)+1)
                for i in range(len(tags)):
                        checksums[i] = self.tables[tags[i]].checkSum

                directory_end = sfntDirectorySize + len(self.tables) * sfntDirectoryEntrySize
                assert directory_end == len(directory)

                checksums[-1] = calcChecksum(directory)
                checksum = numpy.add.reduce(checksums)
                # BiboAfba!
                checksumadjustment = numpy.array(CHECKSUM_MAGIC) - checksum
                # write the checksum to the file
                self.file.seek(self.tables['head'].offset + 8)
                self.file.write(struct.pack(">l", checksumadjustment))


# -- sfnt directory helpers and cruft

sfntDirectoryFormat = """
                > # big endian
                sfntVersion:    4s
                numTables:      H    # number of tables
                searchRange:    H    # (max2 <= numTables)*16
                entrySelector:  H    # log2(max2 <= numTables)
                rangeShift:     H    # numTables*16-searchRange
"""

sfntDirectorySize = sstruct.calcsize(sfntDirectoryFormat)

sfntDirectoryEntryFormat = """
                > # big endian
                tag:            4s
                checkSum:       l
                offset:         l
                length:         l
"""

sfntDirectoryEntrySize = sstruct.calcsize(sfntDirectoryEntryFormat)

class SFNTDirectoryEntry:

        def fromFile(self, file):
                sstruct.unpack(sfntDirectoryEntryFormat,
                                file.read(sfntDirectoryEntrySize), self)

        def fromString(self, str):
                sstruct.unpack(sfntDirectoryEntryFormat, str, self)

        def toString(self):
                return sstruct.pack(sfntDirectoryEntryFormat, self)

        def __repr__(self):
                if hasattr(self, "tag"):
                        return "<SFNTDirectoryEntry '%s' at %x>" % (self.tag, id(self))
                else:
                        return "<SFNTDirectoryEntry at %x>" % id(self)


def calcChecksum(data, start=0):
        """Calculate the checksum for an arbitrary block of data.
        Optionally takes a 'start' argument, which allows you to
        calculate a checksum in chunks by feeding it a previous
        result.

        If the data length is not a multiple of four, it assumes
        it is to be padded with null byte.
        """
        from kiva.fonttools.fontTools import ttLib
        remainder = len(data) % 4
        if remainder:
                data = data + '\0' * (4-remainder)
        a = numpy.fromstring(struct.pack(">l", start) + data, numpy.int32)
        if ttLib.endian != "big":
                a = a.byteswapped()
        return numpy.add.reduce(a)


def maxPowerOfTwo(x):
        """Return the highest exponent of two, so that
        (2 ** exponent) <= x
        """
        exponent = 0
        while x:
                x = x >> 1
                exponent = exponent + 1
        return max(exponent - 1, 0)


def getSearchRange(n):
        """Calculate searchRange, entrySelector, rangeShift for the
        sfnt directory. 'n' is the number of tables.
        """
        # This stuff needs to be stored in the file, because?
        import math
        exponent = maxPowerOfTwo(n)
        searchRange = (2 ** exponent) * 16
        entrySelector = exponent
        rangeShift = n * 16 - searchRange
        return searchRange, entrySelector, rangeShift

