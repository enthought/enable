=====================
Kiva Drawing In-depth
=====================

Kiva State
==========
Kiva is a "stateful" drawing API. What this means is that the graphics context
has a collection of state which affects the results of its drawing actions.
Furthermore, Kiva enables this state to be managed with a stack such that state
can be "pushed" onto the stack before making some temporary changes and then
"popped" off the stack to restore the state to a version which no longer
includes those changes.

State Components
----------------
Here is a list of all the pieces of state tracked by a Kiva graphics context,
along with the methods which operate on them:

* Affine transformation (:py:meth:`translate_ctm`, :py:meth:`rotate_ctm`,
  :py:meth:`scale_ctm`, :py:meth:`concat_ctm`, :py:meth:`set_ctm`,
  :py:meth:`get_ctm`)
* Clipping (:py:meth:`clip_to_rect`, :py:meth:`clip_to_rects`, :py:meth:`clip`,
  :py:meth:`even_odd_clip`)
* Fill color (:py:meth:`set_fill_color`, :py:meth:`get_fill_color`,
  :py:meth:`linear_gradient`, :py:meth:`radial_gradient`)
* Stroke color (:py:meth:`set_stroke_color`, :py:meth:`get_stroke_color`)
* Line width (:py:meth:`set_line_width`)
* Line join style (:py:meth:`set_line_join`)
* Line cap style (:py:meth:`set_line_cap`)
* Line dashing (:py:meth:`set_line_dash`)
* Global transparency (:py:meth:`set_alpha`, :py:meth:`get_alpha`)
* Anti-aliasing (:py:meth:`set_antialias`, :py:meth:`get_antialias`)
* Miter limit (:py:meth:`set_miter_limit`)
* Flatness (:py:meth:`set_flatness`)
* Image interpolation (:py:meth:`set_image_interpolation`, :py:meth:`get_image_interpolation`)
* Text drawing mode (:py:meth:`set_text_drawing_mode`)

Color
-----
Kiva has two colors in its graphics state: stroke color and fill color. Stroke
color is used for the lines in paths when the drawing mode is ``STROKE``,
``FILL_STROKE`` or ``EOF_FILL_STROKE``. Fill color is used for text and for
the enclosed sections of paths when the drawing mode is ``FILL``, ``EOF_FILL``,
``FILL_STROKE``, or ``EOF_FILL_STROKE``. Additionally, the fill color can be
set by the :py:meth:`linear_gradient` and :py:meth:`radial_gradient` methods.

.. note::
   Even though text uses the fill color, text will not be filled with a
   gradient *unless* the text drawing mode is ``TEXT_FILL_STROKE`` and even that
   will only work if the backend supports it.

Color values should always be passed in as 3- or 4- tuples. The order of the
color components is ``(R, G, B[, A])`` and values must be floating point numbers
in the range [0, 1]. Even if a graphics context is not able to draw with alpha
blending, it's still OK to pass a 4 component color value when setting state.

State Stack Management
----------------------
Graphics context instances have two methods for saving and restoring the state,
:py:meth:`save_state` ("push") and :py:meth:`restore_state` ("pop"). That said,
it isn't recommended practice to call the methods directly. Instead, you can
treat the graphics context object as a
`context manager <https://docs.python.org/3/library/stdtypes.html#typecontextmanager>`_
and use the ``with`` keyword to create a block of code where the graphics state
is temporarily modified. Using the context manager approach provides safety from
"temporary" modifications becoming permanent if an uncaught exception is raised
while drawing.

In Enable and Chaco, it is frequently the case that a graphics context instance
will be passed into a method for the purpose of some drawing. Because it is not
reasonable to push the responsibility of state management "up" the call stack,
the onus is on the code making state modifications to do them safely so that
other changes don't leak into other code.

**Well behaved code should take care to only modify graphics state inside a**
``with`` **block**.

Example
-------
.. image:: images/state_ex.png
  :width: 300
  :height: 300

First, the whole example:

.. literalinclude:: state_ex.py
  :linenos:

The first part sets up the default graphics state. Here, that includes a scale
of 2 in X and Y, a translation of (150, 150) which is affected by the
preceeding scale transformation, and some line properties: stroke color, width,
join, and cap:

.. literalinclude:: state_ex.py
  :lines: 7-13
  :linenos:
  :lineno-match:

Then in a loop, we draw twice (the two :py:meth:`stroke_path` calls). The first
draw uses a ``with`` block to temporarily modify the drawing state. It adds more
affine transformations: a rotate and a translate. It also changes some line
properties: stroke color, width, and cap. A rectangle is then added to the 
current path and stroked.

.. literalinclude:: state_ex.py
  :lines: 17-24
  :linenos:
  :lineno-match:

After leaving the first ``with`` block, the state is now restored to its
default. A new ``with`` block is entered and the current transformation matrix
is modified with the same rotation as the first drawing block, but a
*different* translation is applied. The line properties are unchanged
and so use the defaults set at the top.

.. literalinclude:: state_ex.py
  :lines: 26-31
  :linenos:
  :lineno-match:


CompiledPath
============

A path is a collection of geometric objects that can be drawn in a graphics
context with coloring and an affine transformation applied to it. It is the
basic unit of drawing in a graphics context.

Every graphics context instance has a current path which can be manipulated by
the :ref:`kiva_path_functions`. However, some drawing operations are easier to
implement with an independent path instance
(specifically :py:meth:`draw_path_at_points`).

An independent path instance can be created in two ways. The first is via the
:py:meth:`GraphicsContext.get_empty_path` method. The second method is to use
the :class:`CompiledPath` class imported from the backend being used. The
interface of a :class:`CompiledPath` instance is the same as the
:ref:`kiva_path_functions` (modulo :py:meth:`get_empty_path`).

Once you have a path object, it can be drawn by adding it to the graphics
context with the :py:meth:`GraphicsContext.add_path` method (which adds the path
to the current path) and then calling any of the :ref:`kiva_drawing_functions`
which operate on the current path.

For certain backends which support it, the
:py:meth:`GraphicsContext.draw_path_at_points` method can be used to draw a
path object at many different positions with a single function call.

Example
-------
.. image:: images/compiled_path_ex.png
  :width: 300
  :height: 300

.. literalinclude:: compiled_path_ex.py
  :linenos:


Kiva Image Rendering
====================

Drawing images in kiva is accomplished via
:py:meth:`GraphicsContext.draw_image`. A unique feature of drawing images
(relative to path drawing) is that you can apply an arbitrary translation and
scaling to the image without involving the current transformation matrix.

The signature for :py:meth:`draw_image` is straightforward:

.. automethod:: kiva.abstract_graphics_context.AbstractGraphicsContext.draw_image
   :noindex:

The ``image`` object that is passed to :py:meth:`draw_image` can be a numpy
array, a `PIL <https://pillow.readthedocs.io/en/stable/>`_ ``Image`` instance,
or another ``GraphicsContext`` instance of the same backend. If ``image`` is a
numpy array, it is typically converted to a more convenient format via
`PIL.Image.fromarray <https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.fromarray>`_.
Therefore, one must be careful about the expected pixel format of the image. If
your image is rendering with incorrect colors, this might be the problem.
Passing the other allowed versions of ``image`` should give a more consistent
result.

If ``image`` contains an alpha channel and transparent or translucent pixels,
this transparency should also be honored by the destination graphics context.
However, not all backends may support this.

Regarding the ``rect`` argument to :py:meth:`draw_image`, if it is not
specified then the bounding rectangle of the graphics context will be used. As
mentioned before, ``rect`` can be used to apply an arbitrary translation and
scaling to an image. The translation is the x,y position of the rectangle and
the scaling is the ratio of the image's width and height to those of the
rectangle. In every case, ``rect`` will be transformed by the current
transformation matrix.

Special considerations
----------------------
If you only want to draw a subset of an image, you should pass only that subset
to :py:meth:`draw_image`. The Kiva API does not support defining a "source"
rectangle when drawing images, only a "destination".

If drawing images with some scaling applied, one might wish to have control
over the interpolation used when drawing the image. This can be accomplished
with the :py:meth:`set_image_interpolation` method.

.. note::
  :py:meth:`set_image_interpolation` is currently only implemented by the
  ``kiva.agg`` backend. Other backends may have the method, but it is
  effectively a no-op.

Saving images
-------------
One can also save the contents of a graphics context to an image. This is done
via the :py:meth:`save` method:

.. automethod:: kiva.abstract_graphics_context.AbstractGraphicsContext.save
   :noindex:


Kiva Text Rendering
===================

Drawing text in kiva is accomplished via a few methods on 
:class:`GraphicsContext`. There are three basic topics: selecting a font,
measuring the size of rendered text, and drawing the text.

Font Selection
--------------

Font selection for use with the text rendering capabilities of
:class:`GraphicsContext` can be accomplished in a few different ways depending
on the amount of control needed by your drawing code.

Simplest: ``select_font``
~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you're already doing your drawing within an application using traits, you can
use the :class:`kiva.trait_defs.api.KivaFont` trait.

``KivaFont`` traits are initialized with a string which describes the font:
"Times Italic 18", "Courier Bold 10", etc. The *value* of the trait is a
:class:`kiva.fonttools.font.Font` instance which can be passed to the
:py:meth:`GraphicsContext.set_font` method.

**Supported backends**: all backends


``Font`` objects
~~~~~~~~~~~~~~~~

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
--------------

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
------------

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
