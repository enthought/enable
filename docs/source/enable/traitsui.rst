Enable TraitsUI Editors
=======================
To facilitate the inclusion of Enable :class:`~.Component` objects in
`TraitsUI GUIs <https://docs.enthought.com/traitsui>`_, Enable provides
:class:`~.ComponentEditor`.

ComponentEditor
---------------
:class:`~.ComponentEditor` is a fairly simple editor. It only has a few traits
which are of interest to users:

bgcolor
~~~~~~~
``bgcolor`` is a :class:`ColorTrait` which can be used to specify the background
color of the component. The default value is ``"sys_window"``, which may or may
not match the default window background color of the GUI toolkit you are using.

high_resolution
~~~~~~~~~~~~~~~
``high_resolution`` is a boolean which, if True, tells Enable that you would
like your component to take advantage of HiDPI displays if the GUI toolkit
supports it. The default value is True.

size
~~~~
``size`` is a tuple of integers which can be used to specify the initial size of
the component in a GUI. The default value is ``(400, 400)``.
