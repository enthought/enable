Enable Drag and Drop Events
===========================

Enable has support for objects being dropped onto components built on top of
the backend's (and, in practice, PyFace's) support for drag and drop.

All drag-and-drop related events have type ``DragEvent`` and provide the
``x`` and ``y`` coordinates of the event, and the object being dragged or
dropped (if it is available from the backend, otherwise this will be ``None``).

The system generates 3 types of drag and drop events:

``drag_over``
    These events are generated when the user is moving a dragged object over
    the Enable window. Tools or other Interactors which want to indicate that
    they can accept the drag should indicate this by calling the
    ``set_drag_result()`` method of the Enable window indicating the type of
    operation that will be performed on the object (the default is a "copy",
    but other possibilties include "move" or "link", the full list of
    possibilites is found in the appropriate ``constants`` module).  The value
    of the drag result influences the way that the operating system displays
    the dragged objects and cursor whil dragging.

``drag_leave``
    This event is generated when the user drags objects out of the window.

``dropped_on``
    This event is generated when the user releases the mouse button over the
    Enable window while dragging.  Tools or other Interactors should handle
    this event to perform whatever operations need to be performed with the
    dropped objects.

BaseDropTool
------------

As a convenience, there is a ``BaseDropTool`` class which handles most of the
drag and drop interactions for you correctly.  To use this, you need to
subclass and override at least the ``accept_drop`` and ``handle_drop`` methods.

``accept_drop``
    This method is given the position and object instance and should return
    ``True`` if the drop is accepted or ``False`` if it is not.

``handle_drop``
    If the drop is accepted, this method is called with the position and the
    objects, and should perform whatever actions are required with the dropped
    objects.

The behaviour is slightly different between the Wx and Qt backends: Qt provides
a reference to the dragged objects during ``drag_over`` events, and so the
``BaseDropTool`` will call ``accept_drop`` during ``drag_over`` events to give
better feedback about the state of the drag and drop operation; whereas Wx does
not provide that information, so will always indicate to the operating system
that a drop is possible.

The type of drag result returned during ``drag_over`` events is controlled by
the ``default_drag_result`` attribute.

If you want more control over the response to ``drag_over`` events, then you
can additionally override the ``get_drag_result`` method to return one of the
drag result constants dependin on the position and (possibly) the objects
being dragged.  If you want cross-toolkit compatibility, you must handle the
case where the ``get_drag_result`` method is called with the object being
``None``, which indicates that the object is not known yet.
