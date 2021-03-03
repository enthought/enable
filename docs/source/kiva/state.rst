Kiva State
==========
Kiva is a "stateful" drawing API. What this means is that the graphics context
has a collection of state which affect the results of its drawing actions.
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
  :lines: 18-25
  :linenos:
  :lineno-match:

After leaving the ``with`` block, the state is now restored to its default. A
line is drawn from some point 20 units away from the origin -- which was
displaced by the default scale and translate -- to a point which is next to the
rectangle that was drawn a few lines above.

.. literalinclude:: state_ex.py
  :lines: 27-29
  :linenos:
  :lineno-match:
