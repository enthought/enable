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
