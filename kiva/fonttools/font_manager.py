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

A module for finding, managing, and using fonts across platforms.

The design is based on the `W3C Cascading Style Sheet, Level 1 (CSS1)
font specification <http://www.w3.org/TR/1998/REC-CSS2-19980512/>`_.
Future versions may implement the Level 2 or 2.1 specifications.

Authors   : John Hunter <jdhunter@ace.bsd.uchicago.edu>
            Paul Barrett <Barrett@STScI.Edu>
            Michael Droettboom <mdroe@STScI.edu>
Copyright : John Hunter (2004,2005), Paul Barrett (2004,2005)
License   : matplotlib license (PSF compatible)
            The font directory code is from ttfquery,
            see license/LICENSE_TTFQUERY.
"""
import errno
import logging
import os
import pickle
import tempfile
import warnings

from traits.etsconfig.api import ETSConfig

from kiva.fonttools._scan_parse import create_font_list
from kiva.fonttools._scan_sys import scan_system_fonts, scan_user_fonts
from kiva.fonttools._score import (
    score_family, score_size, score_stretch, score_style, score_variant,
    score_weight
)

logger = logging.getLogger(__name__)

# Global singleton of FontManager, cached at the module level.
fontManager = None


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


class FontManager:
    """ The :class:`FontManager` singleton instance is created with a list of
    TrueType fonts based on the font properties: name, style, variant, weight,
    stretch, and size. The :meth:`findfont` method does a nearest neighbor
    search to find the font that most closely matches the specification. If no
    good enough match is found, a default font is returned.
    """
    # Increment this version number whenever the font cache data
    # format or behavior has changed and requires a existing font
    # cache files to be rebuilt.
    __version__ = 9

    def __init__(self, size=None, weight="normal"):
        self._version = self.__version__

        self.__default_weight = weight
        self.default_size = size if size is not None else 12.0

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

        self.ttffiles = scan_system_fonts(paths) + scan_system_fonts()
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

        self.ttflist = create_font_list(self.ttffiles)

        self.afmfiles = scan_system_fonts(
            paths, fontext="afm"
        ) + scan_system_fonts(fontext="afm")
        self.afmlist = create_font_list(self.afmfiles, fontext="afm")
        self.defaultFont["afm"] = None

        self.ttf_lookup_cache = {}
        self.afm_lookup_cache = {}

    def get_default_weight(self):
        """ Return the default font weight.
        """
        return self.__default_weight

    def get_default_size(self):
        """ Return the default font size.
        """
        return self.default_size

    def set_default_weight(self, weight):
        """ Set the default font weight.  The initial value is 'normal'.
        """
        self.__default_weight = weight

    def update_fonts(self, paths):
        """ Update the font lists with new font files.

        The specified ``paths`` will be searched for valid font files and those
        files will have their fonts added to internal collections searched by
        :meth:`findfont`.

        Parameters
        ----------
        filenames : list of str
            A list of font file paths or directory paths.
        """
        afm_paths = scan_user_fonts(paths, fontext="afm")
        ttf_paths = scan_user_fonts(paths, fontext="ttf")

        self.afmlist.extend(create_font_list(afm_paths))
        self.ttflist.extend(create_font_list(ttf_paths))

    def findfont(self, prop, fontext="ttf", directory=None,
                 fallback_to_default=True, rebuild_if_missing=True):
        """ Search the font list for the font that most closely matches
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
        from kiva.fonttools._font_properties import FontProperties

        class FontSpec(object):
            """ An object to represent the return value of findfont().
            """
            def __init__(self, filename, face_index=0):
                self.filename = str(filename)
                self.face_index = face_index

            def __fspath__(self):
                """ Implement the os.PathLike abstract interface.
                """
                return self.filename

            def __repr__(self):
                args = f"{self.filename}, face_index={self.face_index}"
                return f"FontSpec({args})"

        if not isinstance(prop, FontProperties):
            prop = FontProperties(prop)

        fname = prop.get_file()
        if fname is not None:
            logger.debug("findfont returning %s", fname)
            # It's not at all clear where a `FontProperties` instance with
            # `fname` already set would come from. Assume face_index == 0.
            return FontSpec(fname)

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
                score_family(prop.get_family(), font.family) * 10.0
                + score_style(prop.get_style(), font.style)
                + score_variant(prop.get_variant(), font.variant)
                + score_weight(prop.get_weight(), font.weight)
                + score_stretch(prop.get_stretch(), font.stretch)
                + score_size(prop.get_size(), font.size, self.default_size)
            )
            # Lowest score wins
            if score < best_score:
                best_score = score
                best_font = font
            # Exact matches stop the search
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
                return self.findfont(
                    default_prop, fontext, directory,
                    fallback_to_default=False,
                )
            else:
                # This is a hard fail -- we can't find anything reasonable,
                # so just return the vera.ttf
                warnings.warn(
                    "findfont: Could not match %s. Returning %s"
                    % (prop, self.defaultFont[fontext]),
                    UserWarning,
                )
                # Assume this is never a .ttc font, so 0 is ok for face index.
                result = FontSpec(self.defaultFont[fontext])
        else:
            logger.debug(
                "findfont: Matching %s to %s (%s[%d]) with score of %f",
                prop,
                best_font.family,
                best_font.fname,
                best_font.face_index,
                best_score,
            )
            result = FontSpec(best_font.fname, best_font.face_index)

        if not os.path.isfile(result.filename):
            if rebuild_if_missing:
                logger.debug(
                    "findfont: Found a missing font file.  Rebuilding cache."
                )
                _rebuild()
                return default_font_manager().findfont(
                    prop, fontext, directory,
                    fallback_to_default=True,
                    rebuild_if_missing=False,
                )
            else:
                raise ValueError("No valid font could be found")

        if directory is None:
            font_cache[hash(prop)] = result
        return result


# ---------------------------------------------------------------------------
# Utilities

def _get_config_dir():
    """ Return the string representing the configuration dir.
    """
    path = os.path.join(ETSConfig.application_data, "kiva")
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    if not _is_writable_dir(path):
        raise IOError(f"Configuration directory {path} must be writable")

    return path


def _get_font_cache_path():
    """ Return the file path for the font cache to be saved / loaded.

    Returns
    -------
    path : str
        Path to the font cache file.
    """
    return os.path.join(_get_config_dir(), "fontList.cache")


def _is_writable_dir(p):
    """ p is a string pointing to a putative writable dir -- return True p
    is such a string, else False
    """
    if not isinstance(p, str):
        return False

    try:
        with tempfile.TemporaryFile(dir=p) as fp:
            fp.write(b"kiva.test")
        return True
    except OSError:
        pass
    return False


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
        fontManager = _pickle_load(cache_file)
        if (not hasattr(fontManager, "_version")
                or fontManager._version != FontManager.__version__):
            fontManager = _new_font_manager(cache_file)
        else:
            fontManager.default_size = None
            logger.debug("Using fontManager instance from %s", cache_file)
    except Exception:
        fontManager = _new_font_manager(cache_file)

    return fontManager


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
    _pickle_dump(fontManager, cache_file)
    logger.debug("generated new fontManager")
    return fontManager


def _pickle_dump(data, filename):
    """
    Equivalent to pickle.dump(data, open(filename, 'wb'))
    but closes the file to prevent filehandle leakage.
    """
    fh = open(filename, "wb")
    try:
        pickle.dump(data, fh)
    finally:
        fh.close()


def _pickle_load(filename):
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


def _rebuild():
    """ Rebuild the default font manager and cache its content.
    """
    global fontManager
    fontManager = _new_font_manager(_get_font_cache_path())
