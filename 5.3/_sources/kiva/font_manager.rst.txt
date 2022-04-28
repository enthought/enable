.. _kiva_font_management:

Kiva Font Management
====================

Kiva's :class:`FontManager` code originated in the Matplotlib code base. Until
version 5.1.0 it was largely unchanged. From 5.1.0 forward, it has been
refactored to emphasize that it's an internal detail which should be avoided by
3rd party code.

This document aims to describe the structure of the code circa version 5.1.0.

.. note::
    Some Kiva backends don't use the :class:`FontManager` to resolve fonts
    but instead use internal methods. For example the QPainter backend uses
    Qt's font management system to resolve font definitons.

Overview
--------
The basic function of :class:`FontManager` is to provide a :py:meth:`findfont`
method which can be called by the :class:`Font <kiva.fonttools.font.Font>`
class to resolve a font name or font file location on the system where the code
is running.

To accomplish this, it:

1.  Scans the file system for files with known font file extensions (``.ttf``,
    ``.ttc``, ``.otf``, ``.afm``) [:py:func:`scan_system_fonts`]
2.  Examines all the font files identified in the first step and extracts their
    metadata using `fonttools <https://fonttools.readthedocs.io/en/latest/>`_.
    [:py:func:`create_font_database`]
3.  The :class:`FontManager` is then ready to be used.
    :class:`Font <kiva.fonttools.font.Font>` instances call :py:meth:`findfont`
    on the global :class:`FontManager` singleton as needed.

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

1.  If a ``directory`` keyword argument was passed, only fonts whose files are
    children of ``directory`` will be checked. If none match, a default font is
    returned.
2.  If a ``directory`` is not specified, the search is first narrowed by the
    font family (or families) designated by the query. This is possible because
    the match scoring algorithm gives very bad scores to fonts whose family does
    not match the query.
3.  A score is computed for each font using the scoring functions
    [:py:func:`score_family`, :py:func:`score_size`, :py:func:`score_stretch`,
    :py:func:`score_style`, :py:func:`score_variant`, and
    :py:func:`score_weight`]
4.  If the score meets a certain threshold, the matching font is returned.
    Otherwise a default is returned.
5.  If the query was successful, it is added to a local query cache which will
    avoid the scoring process if a matching query is later performed again.

.. _adding_custom_fonts:

Adding Custom Fonts
-------------------

Because font resolution relies heavily on the system that the code is running
on, font matching may not always result in a good match for the desired font.
Kiva ships with a fallback font in case no matching font can be found (this
frequently happens on headless servers with no GUI libraries installed).
However developers may want to ensure that the fonts that they want are always
available by including the font files in the resources that are packaged with
their application.

Kiva provides :py:func:`~kiva.fonttools.app_font.add_application_fonts` as a
mechanism to register additional fonts with the application font resolution
system.  This function should be called early in the application start-up
process with a list of paths of font files to add to the system.  These fonts
will also be added to the Qt and/or Wx font databases as well, as appropriate.

Typical usage might look something like the following::

    from importlib.resources import files
    from kiva.fonttools.api import add_application_fonts

    font_file_1 = files(my_package.resources) / "my_font_1.ttf"
    font_file_2 = files(my_package.resources) / "my_font_2.ttf"

    add_application_fonts([font_file_1, font_file_2])

.. note::
    The font files need to be actual files on the filesystem, they can't be
    stored in zip files.
