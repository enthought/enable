Kiva Text Rendering
===================

Drawing text in kiva is accomplished via a few methods on 
:class:`GraphicsContext`. There are three basic topics: selecting a font,
measuring the size of rendered text, and drawing the text.

Font Selection
++++++++++++++

Font selection for use with the text rendering capabilities of
:class:`GraphicsContext` can be accomplished in a few different ways depending
on the amount of control needed by your drawing code.

Simplest: ``select_font``
-------------------------

The simplest form of font selection is the
:py:meth:`GraphicsContext.select_font` method. The tradeoff for this simplicity
is that you're at the mercy of the backend's font lookup. If your desired font
isn't available from the system you're using, it's not defined what you will end
up with.

``select_font(name, size=12)``

``name`` is the name of the desired font: "Helvetica Regular",
"Futura Medium Italic", etc.

``size`` is the size in points.

**Supported backends**: cairo, celiagg, pdf, ps, qpainter, quartz, svg.


The ``KivaFont`` trait and ``set_font``
---------------------------------------

If you're already doing your drawing within an application using traits, you can
use the :class:`kiva.trait_defs.api.KivaFont` trait.

``KivaFont`` traits are initialized with a string which describes the font:
"Times Italic 18", "Courier Bold 10", etc. The *value* of the trait is a
:class:`kiva.fonttools.font.Font` instance which can be passed to the
:py:meth:`GraphicsContext.set_font` method.

**Supported backends**: all backends


``Font`` objects
----------------

If you don't want to rely on the font description parsing in ``KivaFont``, you
can also manually construct a :class:`kiva.fonttools.font.Font` instance. Once
you have a ``Font`` instance, it can be passed to the
:py:meth:`GraphicsContext.set_font` method.

``Font(face_name="", size=12, family=SWISS, weight=NORMAL, style=NORMAL)``

``face_name`` is the font's name: "Arial", "Webdings", "Verdana", etc.

``size`` is the size in points

``family`` is a constant from :py:mod:`kiva.constants`. Pick from ``DEFAULT``,
``SWISS``, ``ROMAN``, ``MODERN``, ``DECORATIVE``, ``SCRIPT``, or ``TELETYPE``.
If ``face_name`` is empty, the value of ``family`` will be used to select the
desired font.

``weight`` is a constant from :py:mod:`kiva.constants`. Pick from ``NORMAL`` or
``BOLD``.

``style`` is a constant from :py:mod:`kiva.constants`. Pick from ``NORMAL`` or
``ITALIC``.


Measuring Text
++++++++++++++

Before drawing text, one often wants to know what the bounding rectangle of the
rendered text will be so that the text can be positioned correctly. To do this,
the :py:meth:`GraphicsContext.get_text_extent` method is used.

``get_text_extent(text) -> (x, y, width, height)``

``text`` is the string that you want to measure. The currently selected font
will be used, *so it's important to set the font before calling this method.*

The return value is a ``tuple`` which describes a rectangle with its bottom-left
corner at (x, y) and a width and height. The rectangle is relative to the
origin and not affected by the currently set text transform. The bottom of the
rectangle won't always be 0, depending on the font. It might be a negative
number in the situation where glyphs hang below the baseline. In any case,
``y = 0`` is the baseline for the rendered glyphs.

.. note::
   ``get_text_extent`` does not respect endline characters. It is assumed that
   ``text`` describes a single line of text. To render multiple lines, one
   should split the text into individual lines first and then measure and draw
   each line in sequence. A blank line's height should be the same as the
   height of the selected font.


Drawing Text
++++++++++++

Text can be drawn in a graphics context with the
:py:meth:`GraphicsContext.show_text` and
:py:meth:`GraphicsContext.show_text_at_point` methods.

``show_text(text, point=None)``

``show_text_at_point(text, x, y)``

``show_text_at_point`` or ``show_text`` with a ``point=(x, y)`` argument both
do the same thing: Draw a line of text at the given (x, y) coordinate, which
represents the horizontal position of the first glyph and the baseline position,
respectively.

If ``show_text`` is used *without* a ``point`` argument, then the current text
position of the graphics context is used. This position can be set via the
:py:meth:`GraphicsContext.set_text_position` method. Relatedly, the text
position can be retrieved with the :py:meth:`GraphicsContext.get_text_position`
method.

.. note::
   There is also a :py:meth:`GraphicsContext.set_text_matrix` method which
   allows a text-specific affine transform to be set. Unfortunately it's not
   implemented uniformly across backends, so it's recommended not to use it.
