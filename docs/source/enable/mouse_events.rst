Enable Mouse Events
===================

Enable mouse events are represented by the :class:`~.MouseEvent` type and their
event names (which are the suffixes used by
:py:meth:`enable.interactor.Interactor.dispatch`) can be divided into two
groups: mouse clicks and mouse movements. The mouse click events have names
ending in ``_down``, ``_up``, or ``_dclick`` and names beginning with ``left``,
``right``, or ``middle``. This means that Enable only supports three mouse
buttons (plus wheel events).

Event types
-----------

\*_down
~~~~~~~
A mouse button was pressed. Dispatched as ``left_down``, ``right_down``, or
``middle_down``.

\*_up
~~~~~
A mouse button was released. Dispatched as ``left_up``, ``right_up``, or
``middle_up``.

\*_dclick
~~~~~~~~~
A mouse button was double-clicked. Dispatched as ``left_dclick``,
``right_dclick``, or ``middle_dclick``.

move
~~~~
The mouse moved within a component.

enter
~~~~~
The mouse moved into a component's bounds.

leave
~~~~~
The mouse moved out of a component's bounds.

wheel
~~~~~
The mouse wheel moved.

MouseEvent
----------
Below is a listing of the traits on a `MouseEvent` instance.

.. autoclass:: enable.events.MouseEvent
  :members: alt_down, control_down, shift_down, left_down, middle_down, right_down, mouse_wheel, mouse_wheel_axis, mouse_wheel_delta
  :noindex:

