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
