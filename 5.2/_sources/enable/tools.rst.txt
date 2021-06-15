============
Enable Tools
============

Enable ``Tools`` are ``Interator`` subclasses that do not have to have any
visual representation, and which can be dynamically added and removed from
components by adding or removing them from the component's ``tools`` list.
This permits developers to quickly build up complex behaviours from simple,
reproducible parts without having complex inheritance hierarchies.

Basic Tools
-----------

Enable provides a number of basic tools for common interactions.

ButtonTool
~~~~~~~~~~

The :py:class:`ButtonTool` provides basic push-button or checkbox
interactions, depending on how it is configured.  The primary interface it
provides is a :py:attr:`clicked` event which is fired when the user clicks in
the region of the underlying component, or when the :py:meth:`click` method is
called.  The :py:attr:`clicked` event is fired on mouse up.

To get checkbox-style behaviour, set :py:attr:`togglable` to ``True`` and
then every click will invert the :py:attr:`checked` trait.  The toggle state
can also be  changed via the :py:meth:`toggle` method, which does not fire the
:py:attr:`clicked` event when called.  For buttons with multi-state toggles,
subclasses can override the :py:meth:`toggle` method to perform more complex
state changes.

By default, the tool responds to clicks that are within the associated
component, but subclasses can override this behaviour by replacing the
:py:meth:`is_clickable` method with something else.

It will commonly be the case that components or :py:class:`ButtonTool`
subclasses which draw may wish to respond to user interactions by drawing
themselves in a highlighted or selected mode when the mouse is down inside
the button region.  The :py:attr:`down` trait provides this information
conveniently, so that users of the tool can change their drawing state and
request redraws when it changes.

DragTool
~~~~~~~~

The :py:class:`DragTool` is an abstract base class that provides basic
interaction support for draging within Enable.  Many other tools within
Enable and Chaco use it.

HoverTool
~~~~~~~~~

The :py:class:`HoverTool` is a simple tool that calls a callback when the
mouse has been held steadily over the component for a period of time.

MoveTool
~~~~~~~~

A :py:class:`DragTool` subclass that allows a user to move a component around
its container by dragging.

ResizeTool
~~~~~~~~~~

A :py:class:`DragTool` subclass that allows a user to resize a component by
dragging from the edges of the component.

ValueDragTool
~~~~~~~~~~~~~

A :py:class:`DragTool` subclass that allows a drag operation to set an
arbitrary value.


Undo/Redo Support
-----------------

The `enable.tools.pyface` package has a number of modules that provide
classes for working with Pyface's Undo/Redo stack.  This permits Enable
tools to add Commands to the Undo/Redo stack, and provides variants of the
MoveTool and ResizeTool that are undoable.

In addition, a tool is provided which binds keystrokes to send undo and
redo requests to the Pyface UndoManager.

High-Level Tools
~~~~~~~~~~~~~~~~

There are three tools that provide convenient facilities and reference
implementations of interacting with the undo/redo stack.

UndoTool
^^^^^^^^

    The ``UndoTool`` binds keystrokes to undo and redo operations.  The
    ``undo_keys`` and ``redo_keys`` attributes each take a list of ``KeySpec``
    objects which should trigger the relevant operations.  The default
    values bind undo to 'Ctrl+Z' and redo to 'Ctrl+Shift+Z'.

    The ``UndoTool`` must be provided with an ``IUndoManager`` that will
    actually perform the undo and redo operations.

    For example, to bind undo to 'Ctrl+Left arrow', and redo to 'Ctrl+Right
    arrow'::

        undo_tool = UndoTool(
            my_component,
            undo_manager=my_undo_manager,
            undo_keys=[KeySpec('Left', 'control')],
            redo_keys=[KeySpec('Right', 'control')]
        )
        my_component.tools.append(undo_tool)

MoveCommandTool
^^^^^^^^^^^^^^^

    The ``MoveCommandTool`` is a subclass of ``MoveTool`` that by default
    issues a ``MoveCommand`` at the end of every successful drag move.
    A ``MoveCommand`` stores the new and previous position of the
    component so that it can undo and redo the move.  The ``MoveCommandTool``
    needs to be provided with an ``ICommandStack`` instance that it will
    push commands to, but is otherwise identical to the usual ``MoveTool``.

    The command tool has a ``mergeable`` attribute which indicates whether
    subsequent move operations with the same component immediately following
    this one can be merged into one single move operation.

    Typical usage would be something like this::

        move_tool = MoveCommandTool(my_component, command_stack=my_command_stack)
        my_component.tools.append(move_tool)

    Users of the tool can provide a different factory to create appropriate
    ``Command`` instances by setting the ``command`` trait to a callable
    that should expect keyword arguments ``component``, ``data`` (the new
    position), ``previous_position``, and ``mergeable``.

ResizeCommandTool
^^^^^^^^^^^^^^^^^

    The ``ResizeCommandTool`` is a subclass of ``ResizeTool`` that issues
    ``ResizeCommand`` s at the end of every successful drag move.
    A ``ResizeCommand`` stores the new and previous position and bounds of the
    component so that it can undo and redo the resize.  The
    ``ResizeCommandTool`` needs to be provided with an ``ICommandStack``
    instance that it will push commands to, but is otherwise identical to the
    usual ``ResizeTool``.

    The command tool has a ``mergeable`` attribute which indicates whether
    subsequent resize operations with the same component immediately following
    this one can be merged into one single resize operation.

    Typical usage would be something like this::

        move_tool = ResizeTool(my_component, command_stack=my_command_stack)
        my_component.tools.append(move_tool)

    Users of the tool can provide a different factory to create appropriate
    ``Command`` instances by setting the ``command`` trait to a callable
    that should expect keyword arguments ``component``, ``data`` (the new
    rectangle as a tuple ``(x, y, width, height)``), ``previous_rectangle``,
    and ``mergeable``.

Command Classes
~~~~~~~~~~~~~~~

The library provides some useful ``Command`` subclasses that users may want
to create specialized instances or subclass to customize the behaviour
of their applications.  They may also be of use to ``CommandAction`` subclasses
outside of the Enable framework (such as menu items or toolbar buttons) which
want to interact with Enable components.

``ResizeCommand``

    This command handles changing the size of a component.  The constructor
    expects arguments ``component``, ``new_rectangle`` and (optionally)
    ``previous_rectangle``, plus optional additional traits.  If
    ``previous_rectangle`` is not provided, then the component's current
    rectangle is used.

    Instances hold references to the ``Component`` being resized in the
    ``component`` attribute, the new and previous rectangles of the component
    as tuples ``(x, y, width, height)`` in the ``data`` and
    ``previous_rectangle`` attributes, and whether or not subsequent resize
    operations on the same component should be merged together.

    The tool handles the logic of changing the position and bounds of the
    component appropriately, as well as invalidating layout and requesting
    redraws.

    It also provides a default ``name`` attribute of ``Resize `` plus the
    ``component_name`` (which in turn defaults to a more human-readable
    variant of the component's class).  Instances can improve this by
    either supplying a full replacement for the ``name`` attribute, or
    for the ``component_name``.

    Finally, there is a ``move_command`` class method that creates a
    ``ResizeCommand`` that just performs a move and is suitable as the
    command factory of a ``MoveCommandTool``, which allows easy merging
    between resize and move operations, if required for the application.

``MoveCommand``

    This command handles changing the position of a component.  The constructor
    expects arguments ``component``, ``previous_position`` and (optionally)
    ``new_position``, plus optional additional traits.  If ``new_position``
    is not provided, then the component's current position is used.

    Instances hold references to the ``Component`` being moved in the
    ``component`` attribute, the new and previous positions of the component as
    tuples ``(x, y)`` in the ``data`` and ``previous_position`` attributes, and
    whether or not subsequent move operations on the same component should
    be merged together.

    The tool handles the logic of changing the position of the component
    appropriately, as well as invalidating layout and requesting
    redraws.

    It also provides a default ``name`` attribute of ``Move `` plus the
    ``component_name`` (which in turn defaults to a more human-readable
    variant of the component's class).  Instances can improve this by
    either supplying a full replacement for the ``name`` attribute, or
    for the ``component_name``.


Base Classes
~~~~~~~~~~~~

There are two simple base classes of tools that are potentially of use to
authors of new tools.

``BaseUndoTool``

    Tools which need to be able to trigger undo and redo actions, or otherwise
    interact with an undo manager (for example, to set the current command
    stack or clear the command history) can inherit from this class.

    It has an ``undo_manager`` attribute which holds a reference to an
    ``IUndoManager`` and provides convenience methods for ``undo`` and ``redo``
    using the undo manager.

``BaseCommandTool``

    Tools which need to perform undoable actions may want to inherit from this
    class.  It provides a standard ``command_stack`` attribute which
    holds a reference to an ``ICommandStack``.  It also has a ``command``
    callable trait that can be overriden by subclasses to create an
    appropriate command when demanded by the UI.

In addition to these simple base tools, authors of Tools or Actions that
perform undoable operations on Enable or Chaco components may want to make use
of the following ``Command`` subclass:

``ComponentCommand``

    This class is an abstract base class for commands which act on Enable
    ``Components``.  It provides a ``component`` attribute which holds a
    reference to the component that the command should be performed on, and
    a ``component_name`` attribute that can be used to help build the ``name``
    of the ``Command`` to be used in textual representations of the command
    (eg. in menu item labels).

    The default ``component_name`` is just a more human-friendly version of
    the component's class name, with camel-case converted to words.  Users
    are encouraged to override with something even more user-friendly.
