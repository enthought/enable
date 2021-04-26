=========================
Testing Enable Components
=========================

In order to assist in the testing of enable :class:`~.Component` and
:class:`~.Interactor` objects, Enable provides the
:class:`~.EnableTestAssistant` class. This is a mixin class intended to work
along side :class:`python:unittest.TestCase`. Often times in order to
effectively test an enable component and its supported interactivity, one needs
to simulate user interactions such as moving/clicking/dragging the mouse, using
keys, etc. This involves manually creating the corresponding events with
appropriate state and dispatching them appropriately. This can be rather
tedious and :class:`~.EnableTestAssistant` provides a number of helper methods
to greatly simplify the process, making tests both faster/cleaner to write and
easier to read. Furthermore, in test scenarios it is ofen unnecessary to launch
the full application window and as such :class:`~.EnableTestAssistant` provides
a :meth:`~.create_mock_window` method for simply
mocking out the window itself. This allows for specifically testing the
component of interest alone, as is the goal in a unit test. Please see the api
docs (by clicking :class:`~.EnableTestAssistant`) for the full list of methods
available.

.. Todo: Add example test. I was going to refer to an existing test, but none
   of the exissting tests seem very helpful for documentation purposes
