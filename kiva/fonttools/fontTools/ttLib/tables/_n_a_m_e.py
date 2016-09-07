import DefaultTable

import six

import struct
from kiva.fonttools import sstruct
from kiva.fonttools.fontTools.misc.textTools import safeEval
import string
import types

nameRecordFormat = """
                >       # big endian
                platformID:     H
                platEncID:      H
                langID:         H
                nameID:         H
                length:         H
                offset:         H
"""

class table__n_a_m_e(DefaultTable.DefaultTable):

        def decompile(self, data, ttFont):
                format, n, stringoffset = struct.unpack(">HHH", data[:6])
                stringoffset = int(stringoffset)
                stringData = data[stringoffset:]
                data = data[6:stringoffset]
                self.names = []
                for i in range(n):
                        name, data = sstruct.unpack2(nameRecordFormat, data, NameRecord())
                        name.fixLongs()
                        name.string = stringData[name.offset:name.offset+name.length]
                        assert len(name.string) == name.length
                        #if (name.platEncID, name.platformID) in ((0, 0), (1, 3)):
                        #       if len(name.string) % 2:
                        #               print "2-byte string doesn't have even length!"
                        #               print name.__dict__
                        del name.offset, name.length
                        self.names.append(name)

        def compile(self, ttFont):
                if not hasattr(self, "names"):
                        # only happens when there are NO name table entries read
                        # from the TTX file
                        self.names = []
                self.names.sort()  # sort according to the spec; see NameRecord.__cmp__()
                stringData = ""
                format = 0
                n = len(self.names)
                stringoffset = 6 + n * sstruct.calcsize(nameRecordFormat)
                data = struct.pack(">HHH", format, n, stringoffset)
                lastoffset = 0
                done = {}  # remember the data so we can reuse the "pointers"
                for name in self.names:
                        if name.string in done:
                                name.offset, name.length = done[name.string]
                        else:
                                name.offset, name.length = done[name.string] = len(stringData), len(name.string)
                                stringData = stringData + name.string
                        data = data + sstruct.pack(nameRecordFormat, name)
                return data + stringData

        def toXML(self, writer, ttFont):
                for name in self.names:
                        name.toXML(writer, ttFont)

        def fromXML(self, content_tuple, ttFont):
                (name, attrs, content) = content_tuple
                if name != "namerecord":
                        return # ignore unknown tags
                if not hasattr(self, "names"):
                        self.names = []
                name = NameRecord()
                self.names.append(name)
                name.fromXML((name, attrs, content), ttFont)

        def getName(self, nameID, platformID, platEncID, langID=None):
                for namerecord in self.names:
                        if (    namerecord.nameID == nameID and
                                        namerecord.platformID == platformID and
                                        namerecord.platEncID == platEncID):
                                if langID is None or namerecord.langID == langID:
                                        return namerecord
                return None # not found

        def __cmp__(self, other):
                return cmp(self.names, other.names)


class NameRecord:

        def toXML(self, writer, ttFont):
                writer.begintag("namerecord", [
                                ("nameID", self.nameID),
                                ("platformID", self.platformID),
                                ("platEncID", self.platEncID),
                                ("langID", hex(self.langID)),
                                                ])
                writer.newline()
                if self.platformID == 0 or (self.platformID == 3 and self.platEncID in (0, 1)):
                        if len(self.string) % 2:
                                # no, shouldn't happen, but some of the Apple
                                # tools cause this anyway :-(
                                writer.write16bit(self.string + "\0")
                        else:
                                writer.write16bit(self.string)
                else:
                        writer.write8bit(self.string)
                writer.newline()
                writer.endtag("namerecord")
                writer.newline()

        def fromXML(self, content_tuple, ttFont):
                (name, attrs, content) = content_tuple
                self.nameID = safeEval(attrs["nameID"])
                self.platformID = safeEval(attrs["platformID"])
                self.platEncID = safeEval(attrs["platEncID"])
                self.langID =  safeEval(attrs["langID"])
                if self.platformID == 0 or (self.platformID == 3 and self.platEncID in (0, 1)):
                        s = ""
                        for element in content:
                                s = s + element
                        s = six.text_type(s)
                        s = s.strip()
                        self.string = s.encode("utf_16_be")
                else:
                        s = string.strip(string.join(content, ""))
                        self.string = six.text_type(s).encode("latin1")

        def __cmp__(self, other):
                """Compare method, so a list of NameRecords can be sorted
                according to the spec by just sorting it..."""
                selftuple = (self.platformID,
                                self.platEncID,
                                self.langID,
                                self.nameID,
                                self.string)
                othertuple = (other.platformID,
                                other.platEncID,
                                other.langID,
                                other.nameID,
                                other.string)
                return cmp(selftuple, othertuple)

        def __repr__(self):
                return "<NameRecord NameID=%d; PlatformID=%d; LanguageID=%d>" % (
                                self.nameID, self.platformID, self.langID)

        def fixLongs(self):
                """correct effects from bug in Python 1.5.1, where "H"
                returns a Python Long int.
                This has been fixed in Python 1.5.2.
                """
                for attr in dir(self):
                        val = getattr(self, attr)
                        if type(val) == types.LongType:
                                setattr(self, attr, int(val))

