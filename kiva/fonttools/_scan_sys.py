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
This is based heavily on matplotlib's font_manager.py SVN rev 8713
(git commit f8e4c6ce2408044bc89b78b3c72e54deb1999fb5),
but has been modified quite a bit in the decade since it was copied.
####################
"""
import glob
import logging
import os
import subprocess
import sys

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


def scan_user_fonts(fontpaths=None, fontext="ttf"):
    """ Search for fonts in the specified font paths.

    Returns
    -------
    filepaths : list of str
        A list of unique font file paths.
    """
    if fontpaths is None:
        return []

    if isinstance(fontpaths, str):
        fontpaths = [fontpaths]

    fontfiles = set()
    fontexts = _get_fontext_synonyms(fontext)
    for path in fontpaths:
        path = os.path.abspath(path)
        if os.path.isdir(path):
            # For directories, find all the fonts within
            files = []
            for ext in fontexts:
                files.extend(glob.glob(os.path.join(path, "*." + ext)))
                files.extend(glob.glob(os.path.join(path, "*." + ext.upper())))

            for fname in files:
                if os.path.exists(fname) and not os.path.isdir(fname):
                    fontfiles.add(fname)
        elif os.path.exists(path):
            # For files, make sure they have the correct extension
            ext = os.path.splitext(path)[-1][1:].lower()
            if ext in fontexts:
                fontfiles.add(path)

    return sorted(fontfiles)


# ----------------------------------------------------------------------------
# utility funcs

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
