# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
"""
####### NOTE #######
This is based heavily on matplotlib's font_manager.py rev 8713,
but has been modified to not use other matplotlib modules
####################

A module for finding, managing, and using fonts across platforms.

This module provides a single :class:`FontManager` instance that can
be shared across backends and platforms.  The :func:`findfont`
function returns the best TrueType (TTF) font file in the local or
system font path that matches the specified :class:`FontProperties`
instance.  The :class:`FontManager` also handles Adobe Font Metrics
(AFM) font files for use by the PostScript backend.

The design is based on the `W3C Cascading Style Sheet, Level 1 (CSS1)
font specification <http://www.w3.org/TR/1998/REC-CSS2-19980512/>`_.
Future versions may implement the Level 2 or 2.1 specifications.

KNOWN ISSUES

  - documentation
  - font variant is untested
  - font stretch is incomplete
  - font size is incomplete
  - font size_adjust is incomplete
  - default font algorithm needs improvement and testing
  - setWeights function needs improvement
  - 'light' is an invalid weight value, remove it.
  - update_fonts not implemented

Authors   : John Hunter <jdhunter@ace.bsd.uchicago.edu>
            Paul Barrett <Barrett@STScI.Edu>
            Michael Droettboom <mdroe@STScI.edu>
Copyright : John Hunter (2004,2005), Paul Barrett (2004,2005)
License   : matplotlib license (PSF compatible)
            The font directory code is from ttfquery,
            see license/LICENSE_TTFQUERY.
"""
import errno
import glob
import logging
import os
import pickle
import subprocess
import sys
import tempfile
import warnings

from fontTools.ttLib import TTCollection, TTFont, TTLibError

from traits.etsconfig.api import ETSConfig

from . import afm

logger = logging.getLogger(__name__)

font_scalings = {
    "xx-small": 0.579,
    "x-small": 0.694,
    "small": 0.833,
    "medium": 1.0,
    "large": 1.200,
    "x-large": 1.440,
    "xx-large": 1.728,
    "larger": 1.2,
    "smaller": 0.833,
    None: 1.0,
}

stretch_dict = {
    "ultra-condensed": 100,
    "extra-condensed": 200,
    "condensed": 300,
    "semi-condensed": 400,
    "normal": 500,
    "semi-expanded": 600,
    "expanded": 700,
    "extra-expanded": 800,
    "ultra-expanded": 900,
}

weight_dict = {
    "ultralight": 100,
    "light": 200,
    "normal": 400,
    "regular": 400,
    "book": 400,
    "medium": 500,
    "roman": 500,
    "semibold": 600,
    "demibold": 600,
    "demi": 600,
    "bold": 700,
    "heavy": 800,
    "extra bold": 800,
    "black": 900,
}

font_family_aliases = {
    "serif",
    "sans-serif",
    "sans serif",
    "cursive",
    "fantasy",
    "monospace",
    "sans",
    "modern",
}

#  OS Font paths
MSFolders = r"Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders"

MSFontDirectories = [
    r"SOFTWARE\Microsoft\Windows NT\CurrentVersion\Fonts",
    r"SOFTWARE\Microsoft\Windows\CurrentVersion\Fonts",
]

X11FontDirectories = [
    # an old standard installation point
    "/usr/X11R6/lib/X11/fonts/TTF/",
    # here is the new standard location for fonts
    "/usr/share/fonts/",
    # documented as a good place to install new fonts
    "/usr/local/share/fonts/",
    # common application, not really useful
    "/usr/lib/openoffice/share/fonts/truetype/",
]

OSXFontDirectories = [
    "/Library/Fonts/",
    "/Network/Library/Fonts/",
    "/System/Library/Fonts/",
]

home = os.environ.get("HOME")
if home is not None:
    # user fonts on OSX
    path = os.path.join(home, "Library", "Fonts")
    OSXFontDirectories.append(path)
    path = os.path.join(home, ".fonts")
    X11FontDirectories.append(path)

###############################################################################
#  functions to replace those that matplotlib ship in different modules
###############################################################################

preferred_fonts = {
    "fantasy": [
        "Comic Sans MS",
        "Chicago",
        "Charcoal",
        "ImpactWestern",
        "fantasy",
    ],
    "cursive": [
        "Apple Chancery",
        "Textile",
        "Zapf Chancery",
        "Sand",
        "cursive",
    ],
    "monospace": [
        "Bitstream Vera Sans Mono",
        "DejaVu Sans Mono",
        "Andale Mono",
        "Nimbus Mono L",
        "Courier New",
        "Courier",
        "Fixed",
        "Terminal",
        "monospace",
    ],
    "serif": [
        "Bitstream Vera Serif",
        "DejaVu Serif",
        "New Century Schoolbook",
        "Century Schoolbook L",
        "Utopia",
        "ITC Bookman",
        "Bookman",
        "Nimbus Roman No9 L",
        "Times New Roman",
        "Times",
        "Palatino",
        "Charter",
        "serif",
    ],
    "sans-serif": [
        "Bitstream Vera Sans",
        "DejaVu Sans",
        "Lucida Grande",
        "Verdana",
        "Geneva",
        "Lucid",
        "Arial",
        "Helvetica",
        "Avant Garde",
        "sans-serif",
    ],
}


def _is_writable_dir(p):
    """
    p is a string pointing to a putative writable dir -- return True p
    is such a string, else False
    """
    if not isinstance(p, str):
        return False

    try:
        t = tempfile.TemporaryFile(dir=p)
        t.write(b"kiva.test")
        t.close()
    except OSError:
        return False
    else:
        return True


def get_configdir():
    """
    Return the string representing the configuration dir.  If s is the
    special string _default_, use HOME/.kiva.  s must be writable
    """

    p = os.path.join(ETSConfig.application_data, "kiva")
    try:
        os.makedirs(p)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    if not _is_writable_dir(p):
        raise IOError("Configuration directory %s must be writable" % p)
    return p


def decode_prop(prop):
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


def getPropDict(font):
    n = font["name"]
    propdict = {}
    for prop in n.names:
        try:
            if "name" in propdict and "sfnt4" in propdict:
                break
            elif prop.nameID == 1 and "name" not in propdict:
                propdict["name"] = decode_prop(prop.string)
            elif prop.nameID == 4 and "sfnt4" not in propdict:
                propdict["sfnt4"] = decode_prop(prop.string)
        except UnicodeDecodeError:
            continue

    return propdict


###############################################################################
#  matplotlib code below
###############################################################################

synonyms = {
    "ttf": ("ttf", "otf", "ttc"),
    "otf": ("ttf", "otf", "ttc"),
    "ttc": ("ttf", "otf", "ttc"),
    "afm": ("afm",),
}


def get_fontext_synonyms(fontext):
    """
    Return a list of file extensions extensions that are synonyms for
    the given file extension *fileext*.
    """
    return synonyms[fontext]


def win32FontDirectory():
    r"""
    Return the user-specified font directory for Win32.  This is
    looked up from the registry key::

      \\HKEY_CURRENT_USER\Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders\Fonts       # noqa

    If the key is not found, $WINDIR/Fonts will be returned.
    """
    try:
        import winreg
    except ImportError:
        pass  # Fall through to default
    else:
        try:
            user = winreg.OpenKey(winreg.HKEY_CURRENT_USER, MSFolders)
            try:
                try:
                    return winreg.QueryValueEx(user, "Fonts")[0]
                except OSError:
                    pass  # Fall through to default
            finally:
                winreg.CloseKey(user)
        except OSError:
            pass  # Fall through to default
    return os.path.join(os.environ["WINDIR"], "Fonts")


def win32InstalledFonts(directory=None, fontext="ttf"):
    """
    Search for fonts in the specified font directory, or use the
    system directories if none given.  A list of TrueType font
    filenames are returned by default, or AFM fonts if *fontext* ==
    'afm'.
    """

    import winreg

    if directory is None:
        directory = win32FontDirectory()

    fontext = get_fontext_synonyms(fontext)

    key, items = None, {}
    for fontdir in MSFontDirectories:
        try:
            local = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, fontdir)
        except OSError:
            continue

        if not local:
            files = []
            for ext in fontext:
                files.extend(glob.glob(os.path.join(directory, "*." + ext)))
            return files
        try:
            for j in range(winreg.QueryInfoKey(local)[1]):
                try:
                    key, direc, any = winreg.EnumValue(local, j)
                    if not os.path.dirname(direc):
                        direc = os.path.join(directory, direc)
                    direc = os.path.abspath(direc).lower()
                    if os.path.splitext(direc)[1][1:] in fontext:
                        items[direc] = 1
                except EnvironmentError:
                    continue
                except WindowsError:
                    continue

            return list(items.keys())
        finally:
            winreg.CloseKey(local)
    return None


def OSXFontDirectory():
    """
    Return the system font directories for OS X.  This is done by
    starting at the list of hardcoded paths in
    :attr:`OSXFontDirectories` and returning all nested directories
    within them.
    """

    fontpaths = []

    for fontdir in OSXFontDirectories:
        try:
            if os.path.isdir(fontdir):
                fontpaths.append(fontdir)
                for dirpath, dirs, _files in os.walk(fontdir):
                    fontpaths.extend([os.path.join(dirpath, d) for d in dirs])

        except (IOError, OSError, TypeError, ValueError):
            pass
    return fontpaths


def OSXInstalledFonts(directory=None, fontext="ttf"):
    """
    Get list of font files on OS X - ignores font suffix by default.
    """
    if directory is None:
        directory = OSXFontDirectory()

    fontext = get_fontext_synonyms(fontext)

    files = []
    for path in directory:
        if fontext is None:
            files.extend(glob.glob(os.path.join(path, "*")))
        else:
            for ext in fontext:
                files.extend(glob.glob(os.path.join(path, "*." + ext)))
                files.extend(glob.glob(os.path.join(path, "*." + ext.upper())))
    return files


def x11FontDirectory():
    """
    Return the system font directories for X11.  This is done by
    starting at the list of hardcoded paths in
    :attr:`X11FontDirectories` and returning all nested directories
    within them.
    """
    fontpaths = []

    for fontdir in X11FontDirectories:
        try:
            if os.path.isdir(fontdir):
                fontpaths.append(fontdir)
                for dirpath, dirs, _files in os.walk(fontdir):
                    fontpaths.extend([os.path.join(dirpath, d) for d in dirs])

        except (IOError, OSError, TypeError, ValueError):
            pass
    return fontpaths


def get_fontconfig_fonts(fontext="ttf"):
    """
    Grab a list of all the fonts that are being tracked by fontconfig
    by making a system call to ``fc-list``.  This is an easy way to
    grab all of the fonts the user wants to be made available to
    applications, without needing knowing where all of them reside.
    """
    fontext = get_fontext_synonyms(fontext)

    fontfiles = {}
    try:
        cmd = ["fc-list", "", "file"]
        pipe = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        output = pipe.communicate()[0]
    except OSError:
        # Calling fc-list did not work, so we'll just return nothing
        return fontfiles

    output = output.decode("utf8")
    if pipe.returncode == 0:
        for line in output.split("\n"):
            fname = line.split(":")[0]
            if (os.path.splitext(fname)[1][1:] in fontext
                    and os.path.exists(fname)):
                fontfiles[fname] = 1

    return fontfiles


def findSystemFonts(fontpaths=None, fontext="ttf"):
    """
    Search for fonts in the specified font paths.  If no paths are
    given, will use a standard set of system paths, as well as the
    list of fonts tracked by fontconfig if fontconfig is installed and
    available.  A list of TrueType fonts are returned by default with
    AFM fonts as an option.
    """
    fontfiles = {}
    fontexts = get_fontext_synonyms(fontext)

    if fontpaths is None:
        if sys.platform == "win32":
            fontdir = win32FontDirectory()

            fontpaths = [fontdir]
            # now get all installed fonts directly...
            for f in win32InstalledFonts(fontdir):
                base, ext = os.path.splitext(f)
                if len(ext) > 1 and ext[1:].lower() in fontexts:
                    fontfiles[f] = 1
        else:
            # check for OS X & load its fonts if present
            if sys.platform == "darwin":
                fontpaths = []
                for f in OSXInstalledFonts(fontext=fontext):
                    fontfiles[f] = 1
            else:
                # Otherwise, check X11.
                fontpaths = x11FontDirectory()

            for f in get_fontconfig_fonts(fontext):
                fontfiles[f] = 1

    elif isinstance(fontpaths, str):
        fontpaths = [fontpaths]

    for path in fontpaths:
        files = []
        for ext in fontexts:
            files.extend(glob.glob(os.path.join(path, "*." + ext)))
            files.extend(glob.glob(os.path.join(path, "*." + ext.upper())))
        for fname in files:
            abs_path = os.path.abspath(fname)

            # Handle dirs which look like font files, but may contain font
            # files
            if os.path.isdir(abs_path):
                fontpaths.append(abs_path)
            else:
                fontfiles[abs_path] = 1

    return [fname for fname in fontfiles.keys() if os.path.exists(fname)]


def weight_as_number(weight):
    """
    Return the weight property as a numeric value.  String values
    are converted to their corresponding numeric value.
    """
    if isinstance(weight, str):
        try:
            weight = weight_dict[weight.lower()]
        except KeyError:
            weight = 400
    elif weight in range(100, 1000, 100):
        pass
    else:
        raise ValueError("weight not a valid integer")
    return weight


class FontEntry(object):
    """
    A class for storing Font properties.  It is used when populating
    the font lookup dictionary.
    """

    def __init__(self, fname="", name="", style="normal", variant="normal",
                 weight="normal", stretch="normal", size="medium"):
        self.fname = fname
        self.name = name
        self.style = style
        self.variant = variant
        self.weight = weight
        self.stretch = stretch
        try:
            self.size = str(float(size))
        except ValueError:
            self.size = size

    def __repr__(self):
        return "<Font '%s' (%s) %s %s %s %s>" % (
            self.name,
            os.path.basename(self.fname),
            self.style,
            self.variant,
            self.weight,
            self.stretch,
        )


def ttfFontProperty(fpath, font):
    """
    A function for populating the :class:`FontKey` by extracting
    information from the TrueType font file.

    *font* is a :class:`FT2Font` instance.
    """
    props = getPropDict(font)
    name = props.get("name")
    if name is None:
        raise KeyError("No name could be found for: {}".format(fpath))

    #  Styles are: italic, oblique, and normal (default)
    sfnt4 = props.get("sfnt4", "").lower()

    if sfnt4.find("oblique") >= 0:
        style = "oblique"
    elif sfnt4.find("italic") >= 0:
        style = "italic"
    else:
        style = "normal"

    #  Variants are: small-caps and normal (default)

    #  !!!!  Untested
    if name.lower() in ["capitals", "small-caps"]:
        variant = "small-caps"
    else:
        variant = "normal"

    #  Weights are: 100, 200, 300, 400 (normal: default), 500 (medium),
    #    600 (semibold, demibold), 700 (bold), 800 (heavy), 900 (black)
    #    lighter and bolder are also allowed.

    weight = None
    for w in weight_dict.keys():
        if sfnt4.find(w) >= 0:
            weight = w
            break
    if not weight:
        weight = 400
    weight = weight_as_number(weight)

    #  Stretch can be absolute and relative
    #  Absolute stretches are: ultra-condensed, extra-condensed, condensed,
    #    semi-condensed, normal, semi-expanded, expanded, extra-expanded,
    #    and ultra-expanded.
    #  Relative stretches are: wider, narrower
    #  Child value is: inherit

    if (sfnt4.find("narrow") >= 0
            or sfnt4.find("condensed") >= 0
            or sfnt4.find("cond") >= 0):
        stretch = "condensed"
    elif sfnt4.find("demi cond") >= 0:
        stretch = "semi-condensed"
    elif sfnt4.find("wide") >= 0 or sfnt4.find("expanded") >= 0:
        stretch = "expanded"
    else:
        stretch = "normal"

    #  Sizes can be absolute and relative.
    #  Absolute sizes are: xx-small, x-small, small, medium, large, x-large,
    #    and xx-large.
    #  Relative sizes are: larger, smaller
    #  Length value is an absolute font size, e.g. 12pt
    #  Percentage values are in 'em's.  Most robust specification.

    #  !!!!  Incomplete
    size = "scalable"

    return FontEntry(fpath, name, style, variant, weight, stretch, size)


def afmFontProperty(fontpath, font):
    """
    A function for populating a :class:`FontKey` instance by
    extracting information from the AFM font file.

    *font* is a class:`AFM` instance.
    """

    name = font.get_familyname()
    fontname = font.get_fontname().lower()

    #  Styles are: italic, oblique, and normal (default)

    if font.get_angle() != 0 or name.lower().find("italic") >= 0:
        style = "italic"
    elif name.lower().find("oblique") >= 0:
        style = "oblique"
    else:
        style = "normal"

    #  Variants are: small-caps and normal (default)

    # !!!!  Untested
    if name.lower() in ["capitals", "small-caps"]:
        variant = "small-caps"
    else:
        variant = "normal"

    #  Weights are: 100, 200, 300, 400 (normal: default), 500 (medium),
    #    600 (semibold, demibold), 700 (bold), 800 (heavy), 900 (black)
    #    lighter and bolder are also allowed.

    weight = weight_as_number(font.get_weight().lower())

    #  Stretch can be absolute and relative
    #  Absolute stretches are: ultra-condensed, extra-condensed, condensed,
    #    semi-condensed, normal, semi-expanded, expanded, extra-expanded,
    #    and ultra-expanded.
    #  Relative stretches are: wider, narrower
    #  Child value is: inherit
    if (fontname.find("narrow") >= 0
            or fontname.find("condensed") >= 0
            or fontname.find("cond") >= 0):
        stretch = "condensed"
    elif fontname.find("demi cond") >= 0:
        stretch = "semi-condensed"
    elif fontname.find("wide") >= 0 or fontname.find("expanded") >= 0:
        stretch = "expanded"
    else:
        stretch = "normal"

    #  Sizes can be absolute and relative.
    #  Absolute sizes are: xx-small, x-small, small, medium, large, x-large,
    #    and xx-large.
    #  Relative sizes are: larger, smaller
    #  Length value is an absolute font size, e.g. 12pt
    #  Percentage values are in 'em's.  Most robust specification.

    #  All AFM fonts are apparently scalable.

    size = "scalable"

    return FontEntry(fontpath, name, style, variant, weight, stretch, size)


def createFontList(fontfiles, fontext="ttf"):
    """
    A function to create a font lookup list.  The default is to create
    a list of TrueType fonts.  An AFM font list can optionally be
    created.
    """
    # FIXME: This function is particularly difficult to debug
    fontlist = []
    #  Add fonts from list of known font files.
    seen = {}

    font_entry_err_msg = "Could not convert font to FontEntry for file %s"

    for fpath in fontfiles:
        logger.debug("createFontDict %s", fpath)
        fname = os.path.split(fpath)[1]
        if fname in seen:
            continue
        else:
            seen[fname] = 1
        if fontext == "afm":
            try:
                fh = open(fpath, "r")
            except Exception:
                logger.error(
                    "Could not open font file %s", fpath, exc_info=True
                )
                continue
            try:
                try:
                    font = afm.AFM(fh)
                finally:
                    fh.close()
            except RuntimeError:
                logger.error(
                    "Could not parse font file %s", fpath, exc_info=True
                )
                continue
            try:
                prop = afmFontProperty(fpath, font)
            except Exception:
                logger.error(font_entry_err_msg, fpath, exc_info=True)
                continue
        else:
            _, ext = os.path.splitext(fpath)
            try:
                if ext.lower() == ".ttc":
                    with open(fpath, "rb") as f:
                        collection = TTCollection(f)
                        try:
                            props = []
                            for font in collection.fonts:
                                props.append(ttfFontProperty(fpath, font))
                            fontlist.extend(props)
                            continue
                        except Exception:
                            logger.error(
                                font_entry_err_msg, fpath, exc_info=True
                            )
                            continue
                else:
                    font = TTFont(str(fpath))
            except (RuntimeError, TTLibError):
                logger.error(
                    "Could not open font file %s", fpath, exc_info=True
                )
                continue
            except UnicodeError:
                logger.error(
                    "Cannot handle unicode file: %s", fpath, exc_info=True
                )
                continue

            try:
                prop = ttfFontProperty(fpath, font)
            except Exception:
                logger.error(font_entry_err_msg, fpath, exc_info=True)
                continue

        fontlist.append(prop)
    return fontlist


class FontProperties(object):
    """
    A class for storing and manipulating font properties.

    The font properties are those described in the `W3C Cascading
    Style Sheet, Level 1
    <http://www.w3.org/TR/1998/REC-CSS2-19980512/>`_ font
    specification.  The six properties are:

      - family: A list of font names in decreasing order of priority.
        The items may include a generic font family name, either
        'serif', 'sans-serif', 'cursive', 'fantasy', or 'monospace'.

      - style: Either 'normal', 'italic' or 'oblique'.

      - variant: Either 'normal' or 'small-caps'.

      - stretch: A numeric value in the range 0-1000 or one of
        'ultra-condensed', 'extra-condensed', 'condensed',
        'semi-condensed', 'normal', 'semi-expanded', 'expanded',
        'extra-expanded' or 'ultra-expanded'

      - weight: A numeric value in the range 0-1000 or one of
        'ultralight', 'light', 'normal', 'regular', 'book', 'medium',
        'roman', 'semibold', 'demibold', 'demi', 'bold', 'heavy',
        'extra bold', 'black'

      - size: Either an relative value of 'xx-small', 'x-small',
        'small', 'medium', 'large', 'x-large', 'xx-large' or an
        absolute font size, e.g. 12

    Alternatively, a font may be specified using an absolute path to a
    .ttf file, by using the *fname* kwarg.

    The preferred usage of font sizes is to use the relative values,
    e.g.  'large', instead of absolute font sizes, e.g. 12.  This
    approach allows all text sizes to be made larger or smaller based
    on the font manager's default font size.
    """

    def __init__(self, family=None, style=None, variant=None, weight=None,
                 stretch=None, size=None, fname=None, _init=None):
        # if fname is set, it's a hardcoded filename to use
        # _init is used only by copy()

        self._family = None
        self._slant = None
        self._variant = None
        self._weight = None
        self._stretch = None
        self._size = None
        self._file = None

        # This is used only by copy()
        if _init is not None:
            self.__dict__.update(_init.__dict__)
            return

        self.set_family(family)
        self.set_style(style)
        self.set_variant(variant)
        self.set_weight(weight)
        self.set_stretch(stretch)
        self.set_file(fname)
        self.set_size(size)

    def __hash__(self):
        lst = [(k, getattr(self, "get" + k)()) for k in sorted(self.__dict__)]
        return hash(repr(lst))

    def __str__(self):
        attrs = (
            self._family, self._slant, self._variant, self._weight,
            self._stretch, self._size,
        )
        return str(attrs)

    def get_family(self):
        """
        Return a list of font names that comprise the font family.
        """
        return self._family

    def get_name(self):
        """
        Return the name of the font that best matches the font
        properties.
        """
        filename = str(default_font_manager().findfont(self))
        if filename.endswith(".afm"):
            return afm.AFM(open(filename)).get_familyname()

        font = default_font_manager().findfont(self)
        prop_dict = getPropDict(TTFont(str(font)))
        return prop_dict["name"]

    def get_style(self):
        """
        Return the font style.  Values are: 'normal', 'italic' or
        'oblique'.
        """
        return self._slant

    get_slant = get_style

    def get_variant(self):
        """
        Return the font variant.  Values are: 'normal' or
        'small-caps'.
        """
        return self._variant

    def get_weight(self):
        """
        Set the font weight.  Options are: A numeric value in the
        range 0-1000 or one of 'light', 'normal', 'regular', 'book',
        'medium', 'roman', 'semibold', 'demibold', 'demi', 'bold',
        'heavy', 'extra bold', 'black'
        """
        return self._weight

    def get_stretch(self):
        """
        Return the font stretch or width.  Options are: 'ultra-condensed',
        'extra-condensed', 'condensed', 'semi-condensed', 'normal',
        'semi-expanded', 'expanded', 'extra-expanded', 'ultra-expanded'.
        """
        return self._stretch

    def get_size(self):
        """
        Return the font size.
        """
        return self._size

    def get_size_in_points(self):
        if self._size is not None:
            try:
                return float(self._size)
            except ValueError:
                pass
        default_size = default_font_manager().get_default_size()
        return default_size * font_scalings.get(self._size)

    def get_file(self):
        """
        Return the filename of the associated font.
        """
        return self._file

    def set_family(self, family):
        """
        Change the font family.  May be either an alias (generic name
        is CSS parlance), such as: 'serif', 'sans-serif', 'cursive',
        'fantasy', or 'monospace', or a real font name.
        """
        if family is None:
            self._family = None
        else:
            if isinstance(family, bytes):
                family = [family.decode("utf8")]
            elif isinstance(family, str):
                family = [family]
            self._family = family

    set_name = set_family

    def set_style(self, style):
        """
        Set the font style.  Values are: 'normal', 'italic' or
        'oblique'.
        """
        if style not in ("normal", "italic", "oblique", None):
            raise ValueError("style must be normal, italic or oblique")
        self._slant = style

    set_slant = set_style

    def set_variant(self, variant):
        """
        Set the font variant.  Values are: 'normal' or 'small-caps'.
        """
        if variant not in ("normal", "small-caps", None):
            raise ValueError("variant must be normal or small-caps")
        self._variant = variant

    def set_weight(self, weight):
        """
        Set the font weight.  May be either a numeric value in the
        range 0-1000 or one of 'ultralight', 'light', 'normal',
        'regular', 'book', 'medium', 'roman', 'semibold', 'demibold',
        'demi', 'bold', 'heavy', 'extra bold', 'black'
        """
        if weight is not None:
            try:
                weight = int(weight)
                if weight < 0 or weight > 1000:
                    raise ValueError()
            except ValueError:
                if weight not in weight_dict:
                    raise ValueError("weight is invalid")
        self._weight = weight

    def set_stretch(self, stretch):
        """
        Set the font stretch or width.  Options are: 'ultra-condensed',
        'extra-condensed', 'condensed', 'semi-condensed', 'normal',
        'semi-expanded', 'expanded', 'extra-expanded' or
        'ultra-expanded', or a numeric value in the range 0-1000.
        """
        if stretch is not None:
            try:
                stretch = int(stretch)
                if stretch < 0 or stretch > 1000:
                    raise ValueError()
            except ValueError:
                if stretch not in stretch_dict:
                    raise ValueError("stretch is invalid")
        else:
            stretch = 500
        self._stretch = stretch

    def set_size(self, size):
        """
        Set the font size.  Either an relative value of 'xx-small',
        'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'
        or an absolute font size, e.g. 12.
        """
        if size is not None:
            try:
                size = float(size)
            except ValueError:
                if size is not None and size not in font_scalings:
                    raise ValueError("size is invalid")
        self._size = size

    def set_file(self, file):
        """
        Set the filename of the fontfile to use.  In this case, all
        other properties will be ignored.
        """
        self._file = file

    def copy(self):
        """Return a deep copy of self"""
        return FontProperties(_init=self)


def ttfdict_to_fnames(d):
    """
    flatten a ttfdict to all the filenames it contains
    """
    fnames = []
    for named in d.values():
        for styled in named.values():
            for variantd in styled.values():
                for weightd in variantd.values():
                    for stretchd in weightd.values():
                        for fname in stretchd.values():
                            fnames.append(fname)
    return fnames


def pickle_dump(data, filename):
    """
    Equivalent to pickle.dump(data, open(filename, 'wb'))
    but closes the file to prevent filehandle leakage.
    """
    fh = open(filename, "wb")
    try:
        pickle.dump(data, fh)
    finally:
        fh.close()


def pickle_load(filename):
    """
    Equivalent to pickle.load(open(filename, 'rb'))
    but closes the file to prevent filehandle leakage.
    """
    fh = open(filename, "rb")
    try:
        data = pickle.load(fh)
    finally:
        fh.close()
    return data


class FontManager:
    """
    On import, the :class:`FontManager` singleton instance creates a
    list of TrueType fonts based on the font properties: name, style,
    variant, weight, stretch, and size.  The :meth:`findfont` method
    does a nearest neighbor search to find the font that most closely
    matches the specification.  If no good enough match is found, a
    default font is returned.
    """

    # Increment this version number whenever the font cache data
    # format or behavior has changed and requires a existing font
    # cache files to be rebuilt.
    __version__ = 7

    def __init__(self, size=None, weight="normal"):
        self._version = self.__version__

        self.__default_weight = weight
        self.default_size = size

        paths = []

        #  Create list of font paths
        for pathname in ["TTFPATH", "AFMPATH"]:
            if pathname in os.environ:
                ttfpath = os.environ[pathname]
                if ttfpath.find(";") >= 0:  # win32 style
                    paths.extend(ttfpath.split(";"))
                elif ttfpath.find(":") >= 0:  # unix style
                    paths.extend(ttfpath.split(":"))
                else:
                    paths.append(ttfpath)

        logger.debug("font search path %s", str(paths))
        #  Load TrueType fonts and create font dictionary.

        self.ttffiles = findSystemFonts(paths) + findSystemFonts()
        self.defaultFamily = {"ttf": "Bitstream Vera Sans", "afm": "Helvetica"}
        self.defaultFont = {}

        for fname in self.ttffiles:
            logger.debug("trying fontname %s", fname)
            if fname.lower().find("vera.ttf") >= 0:
                self.defaultFont["ttf"] = fname
                break
        else:
            # use anything
            self.defaultFont["ttf"] = self.ttffiles[0]

        self.ttflist = createFontList(self.ttffiles)

        self.afmfiles = findSystemFonts(
            paths, fontext="afm"
        ) + findSystemFonts(fontext="afm")
        self.afmlist = createFontList(self.afmfiles, fontext="afm")
        self.defaultFont["afm"] = None

        self.ttf_lookup_cache = {}
        self.afm_lookup_cache = {}

    def get_default_weight(self):
        """
        Return the default font weight.
        """
        return self.__default_weight

    def get_default_size(self):
        """
        Return the default font size.
        """
        return self.default_size

    def set_default_weight(self, weight):
        """
        Set the default font weight.  The initial value is 'normal'.
        """
        self.__default_weight = weight

    def update_fonts(self, filenames):
        """
        Update the font dictionary with new font files.
        Currently not implemented.
        """
        #  !!!!  Needs implementing
        raise NotImplementedError

    # Each of the scoring functions below should return a value between
    # 0.0 (perfect match) and 1.0 (terrible match)
    def score_family(self, families, family2):
        """
        Returns a match score between the list of font families in
        *families* and the font family name *family2*.

        An exact match anywhere in the list returns 0.0.

        A match by generic font name will return 0.1.

        No match will return 1.0.
        """
        global preferred_fonts

        family2 = family2.lower()
        for i, family1 in enumerate(families):
            family1 = family1.lower()
            if family1 in font_family_aliases:
                if family1 in ("sans", "sans serif", "modern"):
                    family1 = "sans-serif"
                options = preferred_fonts[family1]
                options = [x.lower() for x in options]
                if family2 in options:
                    idx = options.index(family2)
                    return 0.1 * (float(idx) / len(options))
            elif family1 == family2:
                return 0.0
        return 1.0

    def score_style(self, style1, style2):
        """
        Returns a match score between *style1* and *style2*.

        An exact match returns 0.0.

        A match between 'italic' and 'oblique' returns 0.1.

        No match returns 1.0.
        """
        styles = ("italic", "oblique")
        if style1 == style2:
            return 0.0
        elif style1 in styles and style2 in styles:
            return 0.1
        return 1.0

    def score_variant(self, variant1, variant2):
        """
        Returns a match score between *variant1* and *variant2*.

        An exact match returns 0.0, otherwise 1.0.
        """
        if variant1 == variant2:
            return 0.0
        else:
            return 1.0

    def score_stretch(self, stretch1, stretch2):
        """
        Returns a match score between *stretch1* and *stretch2*.

        The result is the absolute value of the difference between the
        CSS numeric values of *stretch1* and *stretch2*, normalized
        between 0.0 and 1.0.
        """
        try:
            stretchval1 = int(stretch1)
        except ValueError:
            stretchval1 = stretch_dict.get(stretch1, 500)
        try:
            stretchval2 = int(stretch2)
        except ValueError:
            stretchval2 = stretch_dict.get(stretch2, 500)
        return abs(stretchval1 - stretchval2) / 1000.0

    def score_weight(self, weight1, weight2):
        """
        Returns a match score between *weight1* and *weight2*.

        The result is the absolute value of the difference between the
        CSS numeric values of *weight1* and *weight2*, normalized
        between 0.0 and 1.0.
        """
        try:
            weightval1 = int(weight1)
        except ValueError:
            weightval1 = weight_dict.get(weight1, 500)
        try:
            weightval2 = int(weight2)
        except ValueError:
            weightval2 = weight_dict.get(weight2, 500)
        return abs(weightval1 - weightval2) / 1000.0

    def score_size(self, size1, size2):
        """
        Returns a match score between *size1* and *size2*.

        If *size2* (the size specified in the font file) is 'scalable', this
        function always returns 0.0, since any font size can be generated.

        Otherwise, the result is the absolute distance between *size1* and
        *size2*, normalized so that the usual range of font sizes (6pt -
        72pt) will lie between 0.0 and 1.0.
        """
        if size2 == "scalable":
            return 0.0
        # Size value should have already been
        try:
            sizeval1 = float(size1)
        except ValueError:
            sizeval1 = self.default_size * font_scalings(size1)
        try:
            sizeval2 = float(size2)
        except ValueError:
            return 1.0
        return abs(sizeval1 - sizeval2) / 72.0

    def findfont(self, prop, fontext="ttf", directory=None,
                 fallback_to_default=True, rebuild_if_missing=True):
        """
        Search the font list for the font that most closely matches
        the :class:`FontProperties` *prop*.

        :meth:`findfont` performs a nearest neighbor search.  Each
        font is given a similarity score to the target font
        properties.  The first font with the highest score is
        returned.  If no matches below a certain threshold are found,
        the default font (usually Vera Sans) is returned.

        `directory`, is specified, will only return fonts from the
        given directory (or subdirectory of that directory).

        The result is cached, so subsequent lookups don't have to
        perform the O(n) nearest neighbor search.

        If `fallback_to_default` is True, will fallback to the default
        font family (usually "Bitstream Vera Sans" or "Helvetica") if
        the first lookup hard-fails.

        See the `W3C Cascading Style Sheet, Level 1
        <http://www.w3.org/TR/1998/REC-CSS2-19980512/>`_ documentation
        for a description of the font finding algorithm.
        """
        if not isinstance(prop, FontProperties):
            prop = FontProperties(prop)
        fname = prop.get_file()
        if fname is not None:
            logger.debug("findfont returning %s", fname)
            return fname

        if fontext == "afm":
            font_cache = self.afm_lookup_cache
            fontlist = self.afmlist
        else:
            font_cache = self.ttf_lookup_cache
            fontlist = self.ttflist

        if directory is None:
            cached = font_cache.get(hash(prop))
            if cached:
                return cached

        best_score = 1e64
        best_font = None

        for font in fontlist:
            fname = font.fname
            if (directory is not None
                    and os.path.commonprefix([fname, directory]) != directory):
                continue
            # Matching family should have highest priority, so it is multiplied
            # by 10.0
            score = (
                self.score_family(prop.get_family(), font.name) * 10.0
                + self.score_style(prop.get_style(), font.style)
                + self.score_variant(prop.get_variant(), font.variant)
                + self.score_weight(prop.get_weight(), font.weight)
                + self.score_stretch(prop.get_stretch(), font.stretch)
                + self.score_size(prop.get_size(), font.size)
            )
            if score < best_score:
                best_score = score
                best_font = font
            if score == 0:
                break

        if best_font is None or best_score >= 10.0:
            if fallback_to_default:
                warnings.warn(
                    "findfont: Font family %s not found. Falling back to %s"
                    % (prop.get_family(), self.defaultFamily[fontext])
                )
                default_prop = prop.copy()
                default_prop.set_family(self.defaultFamily[fontext])
                return self.findfont(default_prop, fontext, directory, False)
            else:
                # This is a hard fail -- we can't find anything reasonable,
                # so just return the vera.ttf
                warnings.warn(
                    "findfont: Could not match %s. Returning %s"
                    % (prop, self.defaultFont[fontext]),
                    UserWarning,
                )
                result = self.defaultFont[fontext]
        else:
            logger.debug(
                "findfont: Matching %s to %s (%s) with score of %f",
                prop,
                best_font.name,
                best_font.fname,
                best_score,
            )
            result = best_font.fname

        if not os.path.isfile(result):
            if rebuild_if_missing:
                logger.debug(
                    "findfont: Found a missing font file.  Rebuilding cache."
                )
                _rebuild()
                return default_font_manager().findfont(
                    prop, fontext, directory, True, False
                )
            else:
                raise ValueError("No valid font could be found")

        if directory is None:
            font_cache[hash(prop)] = result
        return result


_is_opentype_cff_font_cache = {}


def is_opentype_cff_font(filename):
    """
    Returns True if the given font is a Postscript Compact Font Format
    Font embedded in an OpenType wrapper.  Used by the PostScript and
    PDF backends that can not subset these fonts.
    """
    if os.path.splitext(filename)[1].lower() == ".otf":
        result = _is_opentype_cff_font_cache.get(filename)
        if result is None:
            fd = open(filename, "rb")
            tag = fd.read(4)
            fd.close()
            result = tag == "OTTO"
            _is_opentype_cff_font_cache[filename] = result
        return result
    return False


# Global singleton of FontManager, cached at the module level.
fontManager = None


def _get_font_cache_path():
    """ Return the file path for the font cache to be saved / loaded.

    Returns
    -------
    path : str
        Path to the font cache file.
    """
    return os.path.join(get_configdir(), "fontList.cache")


def _rebuild():
    """ Rebuild the default font manager and cache its content.
    """
    global fontManager
    fontManager = _new_font_manager(_get_font_cache_path())


def _new_font_manager(cache_file):
    """ Create a new FontManager (which will reload font files) and immediately
    cache its content with the given file path.

    Parameters
    ----------
    cache_file : str
        Path to the cache to be created.

    Returns
    -------
    font_manager : FontManager
    """
    fontManager = FontManager()
    pickle_dump(fontManager, cache_file)
    logger.debug("generated new fontManager")
    return fontManager


def _load_from_cache_or_rebuild(cache_file):
    """ Load the font manager from the cache and verify it is compatible.
    If the cache is not compatible, rebuild the cache and return the new
    font manager.

    Parameters
    ----------
    cache_file : str
        Path to the cache to be created.

    Returns
    -------
    font_manager : FontManager
    """

    try:
        fontManager = pickle_load(cache_file)
        if (not hasattr(fontManager, "_version")
                or fontManager._version != FontManager.__version__):
            fontManager = _new_font_manager(cache_file)
        else:
            fontManager.default_size = None
            logger.debug("Using fontManager instance from %s", cache_file)
    except Exception:
        fontManager = _new_font_manager(cache_file)

    return fontManager


def default_font_manager():
    """ Return the default font manager, which is a singleton FontManager
    cached in the module.

    Returns
    -------
    font_manager : FontManager
    """
    global fontManager
    if fontManager is None:
        fontManager = _load_from_cache_or_rebuild(_get_font_cache_path())
    return fontManager
