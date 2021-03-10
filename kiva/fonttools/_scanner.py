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

Authors   : John Hunter <jdhunter@ace.bsd.uchicago.edu>
            Paul Barrett <Barrett@STScI.Edu>
            Michael Droettboom <mdroe@STScI.edu>
Copyright : John Hunter (2004,2005), Paul Barrett (2004,2005)
License   : matplotlib license (PSF compatible)
            The font directory code is from ttfquery,
            see license/LICENSE_TTFQUERY.
"""
import glob
import logging
import os
import subprocess
import sys

from fontTools.ttLib import TTCollection, TTFont, TTLibError

from kiva.fonttools import afm
from kiva.fonttools._constants import weight_dict
from kiva.fonttools._util import get_ttf_prop_dict, weight_as_number

logger = logging.getLogger(__name__)
# Error message when fonts fail to load
_FONT_ENTRY_ERR_MSG = "Could not convert font to FontEntry for file %s"

# OS Font paths
MSFolders = r"Software\Microsoft\Windows\CurrentVersion\Explorer\Shell Folders"
MSFontDirectories = [
    r"SOFTWARE\Microsoft\Windows NT\CurrentVersion\Fonts",
    r"SOFTWARE\Microsoft\Windows\CurrentVersion\Fonts",
]
OSXFontDirectories = [
    "/Library/Fonts/",
    "/Network/Library/Fonts/",
    "/System/Library/Fonts/",
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

home = os.environ.get("HOME")
if home is not None:
    # user fonts on OSX
    path = os.path.join(home, "Library", "Fonts")
    OSXFontDirectories.append(path)
    path = os.path.join(home, ".fonts")
    X11FontDirectories.append(path)


class FontEntry(object):
    """ A class for storing Font properties. It is used when populating
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
        fname = os.path.basename(self.fname)
        return (
            f"<FontEntry '{self.name}' ({fname}) {self.style} {self.variant} "
            f"{self.weight} {self.stretch}>"
        )


def create_font_list(fontfiles, fontext="ttf"):
    """ Creates a list of :class`FontEntry` instances from a list of provided
    filepaths.

    The default is to create a list of TrueType fonts. An AFM font list can
    optionally be created.
    """
    # Use a set() to filter out files which were already scanned
    seen = set()

    fontlist = []
    for fpath in fontfiles:
        logger.debug("create_font_list %s", fpath)
        fname = os.path.basename(fpath)
        if fname in seen:
            continue

        seen.add(fname)
        if fontext == "afm":
            fontlist.extend(_build_afm_entries(fpath))
        else:
            fontlist.extend(_build_ttf_entries(fpath))

    return fontlist


def scan_system_fonts(fontpaths=None, fontext="ttf"):
    """ Search for fonts in the specified font paths.

    If no paths are given, will use a standard set of system paths, as
    well as the list of fonts tracked by fontconfig if fontconfig is
    installed and available. A list of TrueType fonts are returned by
    default with AFM fonts as an option.
    """
    fontfiles = set()
    fontexts = _get_fontext_synonyms(fontext)

    if fontpaths is None:
        fontpaths = []
        if sys.platform in ("win32", "cygwin"):
            fontdir = _win32_font_directory()
            fontpaths.append(fontdir)
            # now get all installed fonts directly...
            for fname in _win32_installed_fonts(fontdir):
                ext = os.path.splitext(fname)[-1]
                if ext[1:].lower() in fontexts:
                    fontfiles.add(fname)
        else:
            # check for macOS & load its fonts if present
            if sys.platform == "darwin":
                for fname in _macos_installed_fonts(fontext=fontext):
                    fontfiles.add(fname)
            else:
                # Otherwise, check X11.
                fontpaths = _x11_font_directory()

            for fname in _get_fontconfig_fonts(fontext):
                fontfiles.add(fname)

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
                fontfiles.add(abs_path)

    return [fname for fname in fontfiles if os.path.exists(fname)]


# ----------------------------------------------------------------------------
# Font directory scanning

def _get_fontext_synonyms(fontext):
    """ Return a list of file extensions extensions that are synonyms for
    the given file extension *fileext*.
    """
    synonyms = {
        "ttf": ("ttf", "otf", "ttc"),
        "otf": ("ttf", "otf", "ttc"),
        "ttc": ("ttf", "otf", "ttc"),
        "afm": ("afm",),
    }
    return synonyms.get(fontext, ())


def _get_fontconfig_fonts(fontext="ttf"):
    """ Grab a list of all the fonts that are being tracked by fontconfig
    by making a system call to ``fc-list``.

    This is an easy way to grab all of the fonts the user wants to be
    made available to applications, without needing knowing where all
    of them reside.
    """
    fontext = _get_fontext_synonyms(fontext)
    fontfiles = set()
    try:
        cmd = ["fc-list", "", "file"]
        pipe = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        output = pipe.communicate()[0]
    except OSError:
        # Calling fc-list did not work, so we'll just return nothing
        return fontfiles

    if pipe.returncode == 0:
        output = output.decode("utf8")
        for line in output.split("\n"):
            fname = line.split(":")[0]
            if (os.path.splitext(fname)[1][1:] in fontext
                    and os.path.exists(fname)):
                fontfiles.add(fname)

    return fontfiles


def _macos_font_directory():
    """ Return the system font directories for OS X.

    This is done by starting at the list of hardcoded paths in
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


def _macos_installed_fonts(directory=None, fontext="ttf"):
    """ Get list of font files on OS X - ignores font suffix by default.
    """
    directories = directory
    if directories is None:
        directories = _macos_font_directory()

    fontexts = _get_fontext_synonyms(fontext)
    files = []
    for path in directories:
        if not fontexts:
            files.extend(glob.glob(os.path.join(path, "*")))
        else:
            for ext in fontexts:
                files.extend(glob.glob(os.path.join(path, "*." + ext)))
                files.extend(glob.glob(os.path.join(path, "*." + ext.upper())))
    return files


def _win32_font_directory():
    r""" Return the user-specified font directory for Win32. This is
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


def _win32_installed_fonts(directory=None, fontext="ttf"):
    """ Search for fonts in the specified font directory, or use the system
    directories if none given. A list of TrueType font filenames are returned
    by default, or AFM fonts if *fontext* == 'afm'.
    """
    import winreg

    if directory is None:
        directory = _win32_font_directory()

    fontext = _get_fontext_synonyms(fontext)

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


def _x11_font_directory():
    """ Return the system font directories for X11.

    This is done by starting at the list of hardcoded paths in
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


# ----------------------------------------------------------------------------
# FontEntry Creation

def _build_afm_entries(fpath):
    """ Given the path to an AFM file, return a list of one :class:`FontEntry`
    instance or an empty list if there was an error.
    """
    try:
        fh = open(fpath, "r")
    except OSError:
        logger.error(f"Could not open font file {fpath}", exc_info=True)
        return []

    try:
        font = afm.AFM(fh)
    except RuntimeError:
        logger.error(f"Could not parse font file {fpath}", exc_info=True)
        return []
    finally:
        fh.close()

    try:
        return [_afm_font_property(fpath, font)]
    except Exception:
        logger.error(_FONT_ENTRY_ERR_MSG, fpath, exc_info=True)

    return []


def _build_ttf_entries(fpath):
    """ Given the path to a TTF/TTC file, return a list of :class:`FontEntry`
    instances.
    """
    entries = []

    ext = os.path.splitext(fpath)[-1]
    try:
        with open(fpath, "rb") as fp:
            if ext.lower() == ".ttc":
                collection = TTCollection(fp)
                try:
                    for font in collection.fonts:
                        entries.append(_ttf_font_property(fpath, font))
                except Exception:
                    logger.error(_FONT_ENTRY_ERR_MSG, fpath, exc_info=True)
            else:
                font = TTFont(fp)
                try:
                    entries.append(_ttf_font_property(fpath, font))
                except Exception:
                    logger.error(_FONT_ENTRY_ERR_MSG, fpath, exc_info=True)
    except (RuntimeError, TTLibError):
        logger.error(f"Could not open font file {fpath}", exc_info=True)
    except UnicodeError:
        logger.error(f"Cannot handle unicode file: {fpath}", exc_info=True)

    return entries


def _afm_font_property(fontpath, font):
    """ A function for populating a :class:`FontEntry` instance by
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
    # NOTE: Not sure how many fonts actually have these strings in their name
    variant = "normal"
    for value in ("capitals", "small-caps"):
        if value in name.lower():
            variant = "small-caps"
            break

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
    if fontname.find("demi cond") >= 0:
        stretch = "semi-condensed"
    elif (fontname.find("narrow") >= 0
            or fontname.find("condensed") >= 0
            or fontname.find("cond") >= 0):
        stretch = "condensed"
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


def _ttf_font_property(fpath, font):
    """ A function for populating the :class:`FontEntry` by extracting
    information from the TrueType font file.

    *font* is a :class:`TTFont` instance.
    """
    props = get_ttf_prop_dict(font)
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
    # NOTE: Not sure how many fonts actually have these strings in their name
    variant = "normal"
    for value in ("capitals", "small-caps"):
        if value in name.lower():
            variant = "small-caps"
            break

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
    if sfnt4.find("demi cond") >= 0:
        stretch = "semi-condensed"
    elif (sfnt4.find("narrow") >= 0
            or sfnt4.find("condensed") >= 0
            or sfnt4.find("cond") >= 0):
        stretch = "condensed"
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
