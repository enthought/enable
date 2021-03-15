Top-level Windows
=================
When a component is shown on screen via a GUI toolkit, its :attr:`window` trait
contains an instance of :class:`~.AbstractWindow` which serves as a delegate
between the underlying window system and the component.

For the most part, code doesn't need to interact with the underlying window.
However one common exception is tools which want to set a custom cursor. This
is accomplished via the :py:meth:`set_pointer` method.

AbstractWindow
--------------
The following methods are the public interface of :class:`AbstractWindow`.

.. automethod:: enable.abstract_window.AbstractWindow.get_pointer_position
  :noindex:

.. automethod:: enable.abstract_window.AbstractWindow.redraw
  :noindex:

.. automethod:: enable.abstract_window.AbstractWindow.set_mouse_owner
  :noindex:

.. automethod:: enable.abstract_window.AbstractWindow.set_pointer
  :noindex:

.. automethod:: enable.abstract_window.AbstractWindow.set_tooltip
  :noindex:
