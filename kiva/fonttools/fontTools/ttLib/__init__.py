"""fontTools.ttLib -- a package for dealing with TrueType fonts.

This package offers translators to convert TrueType fonts to Python
objects and vice versa, and additionally from Python to TTX (an XML-based
text format) and vice versa.

Example interactive session:

Python 1.5.2c1 (#43, Mar  9 1999, 13:06:43)  [CW PPC w/GUSI w/MSL]
Copyright 1991-1995 Stichting Mathematisch Centrum, Amsterdam
>>> from kiva.fonttools.fontTools import ttLib
>>> tt = ttLib.TTFont("afont.ttf")
>>> tt['maxp'].numGlyphs
242
>>> tt['OS/2'].achVendID
'B&H\000'
>>> tt['head'].unitsPerEm
2048
>>> tt.saveXML("afont.ttx")
Dumping 'LTSH' table...
Dumping 'OS/2' table...
Dumping 'VDMX' table...
Dumping 'cmap' table...
Dumping 'cvt ' table...
Dumping 'fpgm' table...
Dumping 'glyf' table...
Dumping 'hdmx' table...
Dumping 'head' table...
Dumping 'hhea' table...
Dumping 'hmtx' table...
Dumping 'loca' table...
Dumping 'maxp' table...
Dumping 'name' table...
Dumping 'post' table...
Dumping 'prep' table...
>>> tt2 = ttLib.TTFont()
>>> tt2.importXML("afont.ttx")
>>> tt2['maxp'].numGlyphs
242
>>>

"""

#
# $Id: __init__.py,v 1.36 2002/07/23 16:43:55 jvr Exp $
#

import os
import string
import types


class TTLibError(Exception): pass


class TTFont:

        """The main font object. It manages file input and output, and offers
        a convenient way of accessing tables.
        Tables will be only decompiled when neccesary, ie. when they're actually
        accessed. This means that simple operations can be extremely fast.
        """

        def __init__(self, file=None, res_name_or_index=None,
                        sfntVersion="\000\001\000\000", checkChecksums=0,
                        verbose=0, recalcBBoxes=1):

                """The constructor can be called with a few different arguments.
                When reading a font from disk, 'file' should be either a pathname
                pointing to a file, or a readable file object.

                It we're running on a Macintosh, 'res_name_or_index' maybe an sfnt
                resource name or an sfnt resource index number or zero. The latter
                case will cause TTLib to autodetect whether the file is a flat file
                or a suitcase. (If it's a suitcase, only the first 'sfnt' resource
                will be read!)

                The 'checkChecksums' argument is used to specify how sfnt
                checksums are treated upon reading a file from disk:

                - 0: don't check (default)
                - 1: check, print warnings if a wrong checksum is found
                - 2: check, raise an exception if a wrong checksum is found.

                The TTFont constructor can also be called without a 'file'
                argument: this is the way to create a new empty font.
                In this case you can optionally supply the 'sfntVersion' argument.

                If the recalcBBoxes argument is false, a number of things will *not*
                be recalculated upon save/compile:

                        1. glyph bounding boxes
                        2. maxp font bounding box
                        3. hhea min/max values

                (1) is needed for certain kinds of CJK fonts (ask Werner Lemberg ;-).
                Additionally, upon importing an TTX file, this option cause glyphs
                to be compiled right away. This should reduce memory consumption
                greatly, and therefore should have some impact on the time needed
                to parse/compile large fonts.
                """

                import sfnt
                self.verbose = verbose
                self.recalcBBoxes = recalcBBoxes
                self.tables = {}
                self.reader = None
                if not file:
                        self.sfntVersion = sfntVersion
                        return
                if isinstance(file, basestring):
                        if os.name == "mac" and res_name_or_index is not None:
                                # on the mac, we deal with sfnt resources as well as flat files
                                import macUtils
                                if res_name_or_index == 0:
                                        if macUtils.getSFNTResIndices(file):
                                                # get the first available sfnt font.
                                                file = macUtils.SFNTResourceReader(file, 1)
                                        else:
                                                file = open(file, "rb")
                                else:
                                        file = macUtils.SFNTResourceReader(file, res_name_or_index)
                        else:
                                file = open(file, "rb")
                else:
                        pass # assume "file" is a readable file object
                self.reader = sfnt.SFNTReader(file, checkChecksums)
                self.sfntVersion = self.reader.sfntVersion

        def close(self):
                """If we still have a reader object, close it."""
                if self.reader is not None:
                        self.reader.close()

        def save(self, file, makeSuitcase=0):
                """Save the font to disk. Similarly to the constructor,
                the 'file' argument can be either a pathname or a writable
                file object.

                On the Mac, if makeSuitcase is true, a suitcase (resource fork)
                file will we made instead of a flat .ttf file.
                """
                from kiva.fonttools.fontTools.ttLib import sfnt
                if isinstance(file, basestring):
                        closeStream = 1
                        if os.name == "mac" and makeSuitcase:
                                import macUtils
                                file = macUtils.SFNTResourceWriter(file, self)
                        else:
                                file = open(file, "wb")
                                if os.name == "mac":
                                        import macfs
                                        fss = macfs.FSSpec(file.name)
                                        fss.SetCreatorType('mdos', 'BINA')
                else:
                        # assume "file" is a writable file object
                        closeStream = 0

                tags = self.keys()
                tags.remove("GlyphOrder")
                numTables = len(tags)
                writer = sfnt.SFNTWriter(file, numTables, self.sfntVersion)

                done = []
                for tag in tags:
                        self._writeTable(tag, writer, done)

                writer.close(closeStream)

        def saveXML(self, fileOrPath, progress=None,
                        tables=None, skipTables=None, splitTables=0, disassembleInstructions=1):
                """Export the font as TTX (an XML-based text file), or as a series of text
                files when splitTables is true. In the latter case, the 'fileOrPath'
                argument should be a path to a directory.
                The 'tables' argument must either be false (dump all tables) or a
                list of tables to dump. The 'skipTables' argument may be a list of tables
                to skip, but only when the 'tables' argument is false.
                """
                from kiva.fonttools.fontTools import version
                import xmlWriter

                self.disassembleInstructions = disassembleInstructions
                if not tables:
                        tables = self.keys()
                        if skipTables:
                                for tag in skipTables:
                                        if tag in tables:
                                                tables.remove(tag)
                numTables = len(tables)
                numGlyphs = self['maxp'].numGlyphs
                if progress:
                        progress.set(0, numTables)
                        idlefunc = getattr(progress, "idle", None)
                else:
                        idlefunc = None

                writer = xmlWriter.XMLWriter(fileOrPath, idlefunc=idlefunc)
                writer.begintag("ttFont", sfntVersion=`self.sfntVersion`[1:-1],
                                ttLibVersion=version)
                writer.newline()

                if not splitTables:
                        writer.newline()
                else:
                        # 'fileOrPath' must now be a path
                        path, ext = os.path.splitext(fileOrPath)
                        fileNameTemplate = path + ".%s" + ext

                for i in range(numTables):
                        if progress:
                                progress.set(i)
                        tag = tables[i]
                        if splitTables:
                                tablePath = fileNameTemplate % tagToIdentifier(tag)
                                tableWriter = xmlWriter.XMLWriter(tablePath, idlefunc=idlefunc)
                                tableWriter.begintag("ttFont", ttLibVersion=version)
                                tableWriter.newline()
                                tableWriter.newline()
                                writer.simpletag(tagToXML(tag), src=os.path.basename(tablePath))
                                writer.newline()
                        else:
                                tableWriter = writer
                        self._tableToXML(tableWriter, tag, progress)
                        if splitTables:
                                tableWriter.endtag("ttFont")
                                tableWriter.newline()
                                tableWriter.close()
                if progress:
                        progress.set((i + 1))
                writer.endtag("ttFont")
                writer.newline()
                writer.close()
                if self.verbose:
                        debugmsg("Done dumping TTX")

        def _tableToXML(self, writer, tag, progress):
                if self.has_key(tag):
                        table = self[tag]
                        report = "Dumping '%s' table..." % tag
                else:
                        report = "No '%s' table found." % tag
                if progress:
                        progress.setLabel(report)
                elif self.verbose:
                        debugmsg(report)
                else:
                        print report
                if not self.has_key(tag):
                        return
                xmlTag = tagToXML(tag)
                if hasattr(table, "ERROR"):
                        writer.begintag(xmlTag, ERROR="decompilation error")
                else:
                        writer.begintag(xmlTag)
                writer.newline()
                if tag in ("glyf", "CFF "):
                        table.toXML(writer, self, progress)
                else:
                        table.toXML(writer, self)
                writer.endtag(xmlTag)
                writer.newline()
                writer.newline()

        def importXML(self, file, progress=None):
                """Import a TTX file (an XML-based text format), so as to recreate
                a font object.
                """
                if self.has_key("maxp") and self.has_key("post"):
                        # Make sure the glyph order is loaded, as it otherwise gets
                        # lost if the XML doesn't contain the glyph order, yet does
                        # contain the table which was originally used to extract the
                        # glyph names from (ie. 'post', 'cmap' or 'CFF ').
                        self.getGlyphOrder()
                import xmlImport
                xmlImport.importXML(self, file, progress)

        def isLoaded(self, tag):
                """Return true if the table identified by 'tag' has been
                decompiled and loaded into memory."""
                return self.tables.has_key(tag)

        def has_key(self, tag):
                if self.isLoaded(tag):
                        return 1
                elif self.reader and self.reader.has_key(tag):
                        return 1
                elif tag == "GlyphOrder":
                        return 1
                else:
                        return 0

        def keys(self):
                keys = self.tables.keys()
                if self.reader:
                        for key in self.reader.keys():
                                if key not in keys:
                                        keys.append(key)
                keys.sort()
                if "GlyphOrder" in keys:
                        keys.remove("GlyphOrder")
                return ["GlyphOrder"] + keys

        def __len__(self):
                return len(self.keys())

        def __getitem__(self, tag):
                try:
                        return self.tables[tag]
                except KeyError:
                        if tag == "GlyphOrder":
                                table = GlyphOrder(tag)
                                self.tables[tag] = table
                                return table
                        if self.reader is not None:
                                import traceback
                                if self.verbose:
                                        debugmsg("Reading '%s' table from disk" % tag)
                                data = self.reader[tag]
                                tableClass = getTableClass(tag)
                                table = tableClass(tag)
                                self.tables[tag] = table
                                if self.verbose:
                                        debugmsg("Decompiling '%s' table" % tag)
                                try:
                                        table.decompile(data, self)
                                except "_ _ F O O _ _": # dummy exception to disable exception catching
                                        print "An exception occurred during the decompilation of the '%s' table" % tag
                                        from tables.DefaultTable import DefaultTable
                                        import StringIO
                                        file = StringIO.StringIO()
                                        traceback.print_exc(file=file)
                                        table = DefaultTable(tag)
                                        table.ERROR = file.getvalue()
                                        self.tables[tag] = table
                                        table.decompile(data, self)
                                return table
                        else:
                                raise KeyError, "'%s' table not found" % tag

        def __setitem__(self, tag, table):
                self.tables[tag] = table

        def __delitem__(self, tag):
                if not self.has_key(tag):
                        raise KeyError, "'%s' table not found" % tag
                if self.tables.has_key(tag):
                        del self.tables[tag]
                if self.reader and self.reader.has_key(tag):
                        del self.reader[tag]

        def setGlyphOrder(self, glyphOrder):
                self.glyphOrder = glyphOrder

        def getGlyphOrder(self):
                try:
                        return self.glyphOrder
                except AttributeError:
                        pass
                if self.has_key('CFF '):
                        cff = self['CFF ']
                        if cff.haveGlyphNames():
                                self.glyphOrder = cff.getGlyphOrder()
                        else:
                                # CID-keyed font, use cmap
                                self._getGlyphNamesFromCmap()
                elif self.has_key('post'):
                        # TrueType font
                        glyphOrder = self['post'].getGlyphOrder()
                        if glyphOrder is None:
                                #
                                # No names found in the 'post' table.
                                # Try to create glyph names from the unicode cmap (if available)
                                # in combination with the Adobe Glyph List (AGL).
                                #
                                self._getGlyphNamesFromCmap()
                        else:
                                self.glyphOrder = glyphOrder
                else:
                        self._getGlyphNamesFromCmap()
                return self.glyphOrder

        def _getGlyphNamesFromCmap(self):
                #
                # This is rather convoluted, but then again, it's an interesting problem:
                # - we need to use the unicode values found in the cmap table to
                #   build glyph names (eg. because there is only a minimal post table,
                #   or none at all).
                # - but the cmap parser also needs glyph names to work with...
                # So here's what we do:
                # - make up glyph names based on glyphID
                # - load a temporary cmap table based on those names
                # - extract the unicode values, build the "real" glyph names
                # - unload the temporary cmap table
                #
                if self.isLoaded("cmap"):
                        # Bootstrapping: we're getting called by the cmap parser
                        # itself. This means self.tables['cmap'] contains a partially
                        # loaded cmap, making it impossible to get at a unicode
                        # subtable here. We remove the partially loaded cmap and
                        # restore it later.
                        # This only happens if the cmap table is loaded before any
                        # other table that does f.getGlyphOrder()  or f.getGlyphName().
                        cmapLoading = self.tables['cmap']
                        del self.tables['cmap']
                else:
                        cmapLoading = None
                # Make up glyph names based on glyphID, which will be used by the
                # temporary cmap and by the real cmap in case we don't find a unicode
                # cmap.
                numGlyphs = int(self['maxp'].numGlyphs)
                glyphOrder = [None] * numGlyphs
                glyphOrder[0] = ".notdef"
                for i in range(1, numGlyphs):
                        glyphOrder[i] = "glyph%.5d" % i
                # Set the glyph order, so the cmap parser has something
                # to work with (so we don't get called recursively).
                self.glyphOrder = glyphOrder
                # Get a (new) temporary cmap (based on the just invented names)
                tempcmap = self['cmap'].getcmap(3, 1)
                if tempcmap is not None:
                        # we have a unicode cmap
                        from kiva.fonttools.fontTools import agl
                        cmap = tempcmap.cmap
                        # create a reverse cmap dict
                        reversecmap = {}
                        for unicode, name in cmap.items():
                                reversecmap[name] = unicode
                        allNames = {}
                        for i in range(numGlyphs):
                                tempName = glyphOrder[i]
                                if reversecmap.has_key(tempName):
                                        unicode = reversecmap[tempName]
                                        if agl.UV2AGL.has_key(unicode):
                                                # get name from the Adobe Glyph List
                                                glyphName = agl.UV2AGL[unicode]
                                        else:
                                                # create uni<CODE> name
                                                glyphName = "uni" + string.upper(string.zfill(
                                                                hex(unicode)[2:], 4))
                                        tempName = glyphName
                                        n = 1
                                        while allNames.has_key(tempName):
                                                tempName = glyphName + "#" + `n`
                                                n = n + 1
                                        glyphOrder[i] = tempName
                                        allNames[tempName] = 1
                        # Delete the temporary cmap table from the cache, so it can
                        # be parsed again with the right names.
                        del self.tables['cmap']
                else:
                        pass # no unicode cmap available, stick with the invented names
                self.glyphOrder = glyphOrder
                if cmapLoading:
                        # restore partially loaded cmap, so it can continue loading
                        # using the proper names.
                        self.tables['cmap'] = cmapLoading

        def getGlyphNames(self):
                """Get a list of glyph names, sorted alphabetically."""
                glyphNames = self.getGlyphOrder()[:]
                glyphNames.sort()
                return glyphNames

        def getGlyphNames2(self):
                """Get a list of glyph names, sorted alphabetically,
                but not case sensitive.
                """
                from kiva.fonttools.fontTools.misc import textTools
                return textTools.caselessSort(self.getGlyphOrder())

        def getGlyphName(self, glyphID):
                try:
                        return self.getGlyphOrder()[glyphID]
                except IndexError:
                        # XXX The ??.W8.otf font that ships with OSX uses higher glyphIDs in
                        # the cmap table than there are glyphs. I don't think it's legal...
                        return "glyph%.5d" % glyphID

        def getGlyphID(self, glyphName):
                if not hasattr(self, "_reverseGlyphOrderDict"):
                        self._buildReverseGlyphOrderDict()
                glyphOrder = self.getGlyphOrder()
                d = self._reverseGlyphOrderDict
                if not d.has_key(glyphName):
                        if glyphName in glyphOrder:
                                self._buildReverseGlyphOrderDict()
                                return self.getGlyphID(glyphName)
                        else:
                                raise KeyError, glyphName
                glyphID = d[glyphName]
                if glyphName <> glyphOrder[glyphID]:
                        self._buildReverseGlyphOrderDict()
                        return self.getGlyphID(glyphName)
                return glyphID

        def _buildReverseGlyphOrderDict(self):
                self._reverseGlyphOrderDict = d = {}
                glyphOrder = self.getGlyphOrder()
                for glyphID in range(len(glyphOrder)):
                        d[glyphOrder[glyphID]] = glyphID

        def _writeTable(self, tag, writer, done):
                """Internal helper function for self.save(). Keeps track of
                inter-table dependencies.
                """
                if tag in done:
                        return
                tableClass = getTableClass(tag)
                for masterTable in tableClass.dependencies:
                        if masterTable not in done:
                                if self.has_key(masterTable):
                                        self._writeTable(masterTable, writer, done)
                                else:
                                        done.append(masterTable)
                tabledata = self.getTableData(tag)
                if self.verbose:
                        debugmsg("writing '%s' table to disk" % tag)
                writer[tag] = tabledata
                done.append(tag)

        def getTableData(self, tag):
                """Returns raw table data, whether compiled or directly read from disk.
                """
                if self.isLoaded(tag):
                        if self.verbose:
                                debugmsg("compiling '%s' table" % tag)
                        return self.tables[tag].compile(self)
                elif self.reader and self.reader.has_key(tag):
                        if self.verbose:
                                debugmsg("Reading '%s' table from disk" % tag)
                        return self.reader[tag]
                else:
                        raise KeyError, tag


class GlyphOrder:

        """A pseudo table. The glyph order isn't in the font as a separate
        table, but it's nice to present it as such in the TTX format.
        """

        def __init__(self, tag):
                pass

        def toXML(self, writer, ttFont):
                glyphOrder = ttFont.getGlyphOrder()
                writer.comment("The 'id' attribute is only for humans; "
                                "it is ignored when parsed.")
                writer.newline()
                for i in range(len(glyphOrder)):
                        glyphName = glyphOrder[i]
                        writer.simpletag("GlyphID", id=i, name=glyphName)
                        writer.newline()

        def fromXML(self, (name, attrs, content), ttFont):
                if not hasattr(self, "glyphOrder"):
                        self.glyphOrder = []
                        ttFont.setGlyphOrder(self.glyphOrder)
                if name == "GlyphID":
                        self.glyphOrder.append(attrs["name"])


def _test_endianness():
        """Test the endianness of the machine. This is crucial to know
        since TrueType data is always big endian, even on little endian
        machines. There are quite a few situations where we explicitly
        need to swap some bytes.
        """
        import struct
        data = struct.pack("h", 0x01)
        if data == b"\000\001":
                return "big"
        elif data == b"\001\000":
                return "little"
        else:
                assert 0, "endian confusion!"

endian = _test_endianness()


def getTableModule(tag):
        """Fetch the packer/unpacker module for a table.
        Return None when no module is found.
        """
        import tables
        pyTag = tagToIdentifier(tag)
        try:
                module = __import__("kiva.fonttools.fontTools.ttLib.tables." + pyTag)
        except ImportError:
                return None
        else:
                return getattr(tables, pyTag)


def getTableClass(tag):
        """Fetch the packer/unpacker class for a table.
        Return None when no class is found.
        """
        module = getTableModule(tag)
        if module is None:
                from tables.DefaultTable import DefaultTable
                return DefaultTable
        pyTag = tagToIdentifier(tag)
        tableClass = getattr(module, "table_" + pyTag)
        return tableClass


def newTable(tag):
        """Return a new instance of a table."""
        tableClass = getTableClass(tag)
        return tableClass(tag)


def _escapechar(c):
        """Helper function for tagToIdentifier()"""
        import re
        if re.match("[a-z0-9]", c):
                return "_" + c
        elif re.match("[A-Z]", c):
                return c + "_"
        else:
                return hex(ord(c))[2:]


def tagToIdentifier(tag):
        """Convert a table tag to a valid (but UGLY) python identifier,
        as well as a filename that's guaranteed to be unique even on a
        caseless file system. Each character is mapped to two characters.
        Lowercase letters get an underscore before the letter, uppercase
        letters get an underscore after the letter. Trailing spaces are
        trimmed. Illegal characters are escaped as two hex bytes. If the
        result starts with a number (as the result of a hex escape), an
        extra underscore is prepended. Examples::

                'glyf' -> '_g_l_y_f'
                'cvt ' -> '_c_v_t'
                'OS/2' -> 'O_S_2f_2'
        """
        import re
        if tag == "GlyphOrder":
                return tag
        assert len(tag) == 4, "tag should be 4 characters long"
        while len(tag) > 1 and tag[-1] == ' ':
                tag = tag[:-1]
        ident = ""
        for c in tag:
                ident = ident + _escapechar(c)
        if re.match("[0-9]", ident):
                ident = "_" + ident
        return ident


def identifierToTag(ident):
        """the opposite of tagToIdentifier()"""
        if ident == "GlyphOrder":
                return ident
        if len(ident) % 2 and ident[0] == "_":
                ident = ident[1:]
        assert not (len(ident) % 2)
        tag = ""
        for i in range(0, len(ident), 2):
                if ident[i] == "_":
                        tag = tag + ident[i+1]
                elif ident[i+1] == "_":
                        tag = tag + ident[i]
                else:
                        # assume hex
                        tag = tag + chr(string.atoi(ident[i:i+2], 16))
        # append trailing spaces
        tag = tag + (4 - len(tag)) * ' '
        return tag


def tagToXML(tag):
        """Similarly to tagToIdentifier(), this converts a TT tag
        to a valid XML element name. Since XML element names are
        case sensitive, this is a fairly simple/readable translation.
        """
        import re
        if tag == "OS/2":
                return "OS_2"
        elif tag == "GlyphOrder":
                return "GlyphOrder"
        if re.match("[A-Za-z_][A-Za-z_0-9]* *$", tag):
                return string.strip(tag)
        else:
                return tagToIdentifier(tag)


def xmlToTag(tag):
        """The opposite of tagToXML()"""
        if tag == "OS_2":
                return "OS/2"
        if len(tag) == 8:
                return identifierToTag(tag)
        else:
                return tag + " " * (4 - len(tag))
        return tag


def debugmsg(msg):
        import time
        print msg + time.strftime("  (%H:%M:%S)", time.localtime(time.time()))

