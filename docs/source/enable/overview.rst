======
Enable
======

Enable is a library which provides an interactive 2D canvas, upon which one can
build `interactive plots <https://docs.enthought.com/chaco>`_ or other
applications. It is analogous to the
`HTML5 \<canvas\> element <https://developer.mozilla.org/en-US/docs/Web/API/Canvas_API>`_
and provides a similar 2D drawing interface via :ref:`kiva_overview`.

Enable is a foundational layer of the Chaco plotting library, and is what is
responsible for handling user interaction and interfacing with the underlying
GUI toolkit. Developers interested in writing code that implements tools for
Chaco will need to be familiar with the Enable API.


Enable Concepts
---------------

Fundamentally an Enable application is made up of a few key parts. Most
important is the :class:`Component`, which is the interactive canvas which a
user interacts with. Added to the root component are any number of :class:`Tool`
instances which handle the event dispatches. A root component might also contain
one or more child component instances which are held in the :attr:`overlays` or
:attr:`underlays` traits. These child components often handle display of
individual tools, depending on need.


Component
---------

:class:`Component` is the most important object in Enable, the center of
everything. It represents a visual component. It both draws a screen object, and
receives input for it (keyboard, mouse, and multitouch events).

Basic traits of Component include:

* :attr:`visible`: Whether it's visible
* :attr:`invisible_layout`: Whether it uses space even when not visible (by
  default, invisible objects don't take up space in layout)

Padding
~~~~~~~

Layout in Enable uses padding, similar to CSS. In Chaco, it's used for things
around the edges of plot, like labels and tick marks that extend outside the
main plot area.

* :attr:`fill_padding`: Whether the background color fills the padding area as
  well as the main area of the component.
* :attr:`padding_left`
* :attr:`padding_right`
* :attr:`padding_top`
* :attr:`padding_bottom`
* :attr:`padding`: Sets or gets all 4 padding size traits at once
* :attr:`hpadding`: Read-only convenience property for the total amount of
  horizontal padding
* :attr:`vpadding`: Read-only convenience property for the total amount of
  vertical padding
* :attr:`padding_accepts_focus`: Whether the component responds to mouse events
  over the padding area

Parent Classes
~~~~~~~~~~~~~~

:class:`Component` subclasses both :class:`CoordinateBox` (for drawing) and
:class:`Interactor` (for input). :class:`CoordinateBox` has :attr:`position` and
:attr:`bounds` traits, and some secondary attributes for convenience: :attr:`x`,
:attr:`y`, :attr:`x2`, :attr:`y2`, :attr:`width`, :attr:`height`.
:class:`Interactor` mixes in responses for event types. You can subclass one of
these classes if you want only its capabilities. For example, if you want
something that doesn't draw but does respond to events, subclass
:class:`Interactor` (e.g., a tool).

:class:`Interactor` defines common traits for screen interaction, including:

* :attr:`pointer`: The cursor shape when the interactor is active
* :attr:`event_state`: The object's event state, used for event dispatch

Container
~~~~~~~~~

All components have a :class:`Container`. They can only have a single 
container. One component can't be contained by two objects.

Whenever you request a component to redraw itself, it actually requests its
container to redraw it, and a whole chain goes all the up to the top-level
window.

Top-level Window
~~~~~~~~~~~~~~~~

A component also has a reference to the top-level window. This window serves as
a bridge between the OS and GUI toolkit. The :attr:`window` trait delegates all
the way up the containment chain to the top-level component, which has an actual
reference to the actual window.

The reference to the window is useful because Enable doesn't make calls directly
to the GUI toolkit. Rather, it asks the window to do things for it, such as
creating a context menu.

Event Dispatch
~~~~~~~~~~~~~~

The key methods of :class:`Interactor` are :meth:`dispatch` and
:meth:`\_dispatch_stateful_event`. There's a complex method resolution that
occurs beween :class:`Interactor`, :class:`Component`, :class:`Container`
(which is a subclass of :class:`Component`), and the Chaco-based subclasses of
Enable :class:`Component` and :class:`Container`.

When a component gets an event, it tries to handle it in a standard way, which
is to dispatch to:

1. its active tool
2. its overlays
3. itself, so that any event handler methods on itself get called
4. its underlays
5. its listener tools

That logic is in :class:`Component`, in the :meth:`\_new_dispatch` method, which
is called from :meth:`Component.dispatch` (:meth:`\_old_dispatch` is still
being used by Chaco). If any of these handlers sets event.handled to True, event
propagation stops. If an event gets as far as the listener tools, then all of
them get the event.

.. note::

  The notion of an active tool is not used in current code, just older client
  code. Experience has shown that the notion of a tool promoting itself to be
  the "active" tool isn't really useful, because usually the tools need to
  interact with each other. For newer tools, such as Pan, Zoom, or !DragZoom,
  when the user starts interacting with a tool, that tool calls capture_mouse()
  at the window level, and then all mouse events go to that tool, circumventing
  the entire dispatch() mechanism.

The event handlers that :class:`Component` dispatches to are of the form
:samp:`{event_state}{event_suffix}`, where *event_suffix* corresponds to the
actual kind of event that happened, e.g., :obj:`left_down`, :obj:`left_up`,
:obj:`left_dclick`, etc. Most objects default to having just a single event
state, which is the "normal" event state. To make an Enable component that
handled a left-click, you could subclass :class:`Component`, and implement
:meth:`normal_left_down` or :meth:`normal_left_up`. The signature for handler
methods is just one parameter, which is an event object that is an instance of
(a subclass of) :class:`BasicEvent`. Some subclasses of :class:`BasicEvent`
include :class:`MouseEvent`, :class:`DragEvent`, :class:`KeyEvent`, and
:class:`BlobEvent` (for multitouch). It's fairly easy to extend this event
system with new kinds of events and new suffixes (as was done for multitouch). A
disadvantage is that you don't necessarily get feedback when you misspell an
event handler method name in its definition.

.. note::

  This scheme is difficult to implement when the number of states and events
  gets large. There's nothing to tell you if you've forgotten to implement one
  of the possible combinations.

If an interactor transforms an event, then it has to return the full
transformation that it applies to the event.

When an event comes in, it has a reference to the GUI toolkit window that the
event came from. Lots of code calls methods on :obj:`event.window` to get the
window to do things, such as set a tooltip or create a context menu. That is the
correct thing to do, because it's possible for there to be two windows showing
the same underlying component, so responses to events in a window should only
happen in that window. When the user generates an event, that event propagates
down the containment stack and things happen in response; a draw or update
doesn't actually happen until the next :meth:`paint`. By that time, the
component no longer has a reference to the event or the event's window; instead
it uses its own reference to the window, :obj:`self.window`.

Coordinate Systems
~~~~~~~~~~~~~~~~~~

Every component has :attr:`x` and :attr:`y` traits from :class:`CoordinateBox`.
These are positions relative to the component's parent container. When a
container dispatches events, or loops over its children to draw, it transforms
the coordinate system, so that as far as its children are concerned, the events
are relative to the lower-left corner of the parent container. Objects don't
have to be bounded, but they do have to have an origin. The component is going
to give coordinates to the :class:`GraphicsContext` in its own coordinate
system, and the container is responsible for offsetting the GC, and setting up
the transform correctly. Likewise, when a component gets an event, it expects
that event to be in the coordinate system of its parent container.

.. note::

  This introduces some complexity in trying to handle mouse event capture. If a
  tool or component captures the mouse, the top-level window has no idea what
  the coordinate system of that object is. It has to be able to ask an event,
  "give me your total transformation up to this point", and then apply that
  transformation to all subsequent events. Programmers using Chaco or Enable
  don't usually have to think about this, but the interactor does have to be
  able to do it. Containers implement this, so if you're just writing a standard
  component, you don't have to worry about it.

Viewports
~~~~~~~~~

A component can have a list of viewports, which are views onto the component.
Currently, this is used for the canvas, and for geophysical plotting. You could
use it for something like a magnifying-glass view of a portion of a component or
plot without duplicating it.


Layout
~~~~~~
Containers are the sizers that do layout. Components within containers can
declare that they are resizable, for example, but that doesn't matter if
the container they are in doesn't do layout.

The basic traits on :class:`Component` for layout are :attr:`resizable`,
:attr:`aspect_ratio`, :attr:`auto_center`. For the :attr:`resizable` trait,
you can specify which directions the component is resizable in. Components
also have lists of overlays and underlays.

You can get access to the actual bounds of the component, including its
padding with the :samp:`outer_{name}` attributes. Those also take into account
the thickness of any border around the component.

For more control over layout, there is a
:ref:`constraints-based layout<constraints-layout>` system available.

Rendering
~~~~~~~~~

Every component can have several layers:

* background
* image (Chaco only, not Enable)
* underlay
* main layer (the actual component)
* overlay

These are defined by DEFAULT_DRAWING_ORDER, and stored in the
:attr:`drawing_order` trait.

Complexity arises when you have multiple components in a container: How do
their layers affect each other? Do you want the "overlay" layer of a component
to draw on top of all components? Do you want the "background" elements
to be behind everything else?

This is resolved by the :attr:`unified_draw` trait. If it is False (the
default), the corresponding layers of all components are drawn in sequence. The
container is responsible for calling the components to draw their layers in
the correct sequence. If it is True, then all layers of the component are drawn
in strict sequence. The point is the overall sequence at which a component
with ``unified_draw==True`` is drawn is determined by its :attr:`draw_layer`
trait, which by default is 'mainlayer'.

For example, if you want a plot to act as an overlay, you could set
``unified_draw==True`` and ``draw_layer=='overlay'``. These values tell the
container to render the component when it gets to the 'overlay' layer.

Set :attr:`overlay_border` to True if you want the border to draw as part of
the overlay; otherwise it draws as part of the background. By default,
the border is drawn just inside the plot area; set :attr:`inset_border` to
False to draw it just outside the plot area.

Backbuffer
^^^^^^^^^^

A backbuffer provides the ability to render into an offscreen buffer, which is
blitted on every draw, until it is invalidated. Various traits such as
:attr:`use_backbuffer` and :attr:`backbuffer_padding` control the behavior of
the backbuffer. A backbuffer is used for non-OpenGL backends, such as `agg`
and on OS X. If :attr:`use_backbuffer` is False, a backbuffer is never used,
even if a backbuffer is referenced by a component.

Users typically subclass Chaco :class:`PlotComponent`, but may need features
from Enable :class:`Component`.


Container
---------

:class:`Container` is a subclass of Enable :class:`Component`. Containers can be
nested. Containers are responsible for event dispatch, draw dispatch, and
layout. Containers override a lot of Component methods, so that they behave more
like containers than plain components do.
