Kiva Font Management
====================

Kiva's :class:`FontManager` code originated in the Matplotlib code base. Until
version 5.1.0 it was largely unchanged. From 5.1.0 forward, it has been
refactored to emphasize that it's an internal detail which should be avoided by
3rd party code.

This document aims to describe the structure of the code circa version 5.1.0.

Overview
--------
The basic function of :class:`FontManager` is to provide a :py:meth:`findfont`
method which can be called by the :class:`~.Font` class to resolve a font name
or font file location on the system where the code is running.

To accomplish this, it:

1. Scans the file system for files with known font file extensions (``.ttf``,
   ``.ttc``, ``.otf``, ``.afm``) [:py:func:`scan_system_fonts`]
2. Examines all the font files identified in the first step and extracts their
   metadata using `fonttools <https://fonttools.readthedocs.io/en/latest/>`_.
   [:py:func:`create_font_database`]
3. The :class:`FontManager` is then ready to be used. :class:`~.Font` instances
   call :py:meth:`findfont` on the global :class:`FontManager` singleton as
   needed.

Because scanning a system for available fonts is quite an expensive operation,
:class:`FontManager` stores a
`pickled <https://docs.python.org/3/library/pickle.html>`_ copy of its global
singleton in a cache file. And because ``pickle`` is notoriously brittle, the
font manager has a :py:attr:`__version__` attribute must be incremented any
time the attributes or their classes (:class:`FontEntry`, :class:`FontDatabase`,
etc.) change. The :class:`FontManager` singleton is loaded on-demand by the
first call to :py:func:`default_font_manager`.

Font Resolution
---------------
The :py:meth:`findfont` method uses a somewhat complex process for finding the
best match to a given font query.

1. If a ``directory`` keyword argument was passed, only fonts whose files are
   children of ``directory`` will be checked. If none match, a default font is
   returned.
2. If a ``directory`` is not specified, the search is first narrowed by the
   font family (or families) designated by the query. This is possible because
   the match scoring algorithm gives very bad scores to fonts whose family does
   not match the query.
3. A score is computed for each font using the scoring functions
   [:py:func:`score_family`, :py:func:`score_size`, :py:func:`score_stretch`,
   :py:func:`score_style`, :py:func:`score_variant`, and
   :py:func:`score_weight`]
4. If the score meets a certain threshold, the matching font is returned.
   Otherwise a default is returned.
5. If the query was successful, it is added to a local query cache which will
   avoid the scoring process if a matching query is later performed again.
