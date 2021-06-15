Enable Custom Traits
====================
Enable defines several `trait types <https://docs.enthought.com/traits>`_ which
can be used to simplify the generation of values to pass to various APIs
throughout Enable (and Kiva).


bounds_trait
------------
:class:`~.bounds_trait` represents the bounds of an object. It is a list of two
values: width and height.

ColorTrait
----------
:class:`ColorTrait` represents an RGBA color. One can assign either tuples
containing RGBA colors (each component in the range [0, 1]) or a string in
the form of an HTML color name ("blue" or "#0000FF").

font_trait
----------
:class:`~.font_trait` is a synonym for :class:`kiva.trait_defs.api.KivaFont`.
The trait maps a font-description string to a valid :class:`kiva.fonttools.Font`
instance which can be passed to :py:meth:`AbstractGraphicsContext.set_font`

LineStyle
---------
:class:`~.LineStyle` represents the dash style of a line drawn with Kiva.
Allowed values are "solid", "dot dash", "dash", "dot", or "long dash".

MarkerTrait
-----------
:class:`~.MarkerTrait` represents a marker which can be drawn by Kiva. Allowed
values are "square", "circle", "triangle", "inverted_triangle", "left_triangle",
"right_triangle", "pentagon", "hexagon", "hexagon2", "plus", "cross", "star",
"cross_plus", "diamond", "dot", or "pixel".

Pointer
-------
:class:`~.Pointer` represents the style of a mouse pointer on screen. Allowed
values are "arrow", "right arrow", "blank", "bullseye", "char", "cross", "hand",
"ibeam", "left button", "magnifier", "middle button", "no entry", "paint brush",
"pencil", "point left", "point right", "question arrow", "right button",
"size top", "size bottom", "size left", "size right", "size top right",
"size bottom left", "size top left", "size bottom right", "sizing", "spray can",
"wait", "watch", or "arrow wait".
