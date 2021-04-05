.. _toolkit-selection:

Enable Toolkit Selection
========================

One of the more complicated aspects of Enable is the way in which it selects
which widget toolkit (Qt, Wx, etc.) to use when constructing a :class:`Window`
instance. Further, there is also a second choice of which kiva
:class:`GraphicsContext` implementation to use along with the selected widget
toolkit.

There are of course defaults for both toolkit and kiva backend, which might even
depend on the packages available in the Python environment where code is
running. For most users, these defaults are sufficient and stable over time. For
users that have specific needs, the widget toolkit and kiva backend can be
explicitly specified.


ETSConfig
---------

To control toolkit selection directly from application code, one must use
:class:`~traits.etsconfig.etsconfig.ETSConfig`. Specifically, the
``ETSConfig.toolkit`` trait should be set.

.. note::
  ``ETSConfig.toolkit`` must be set before importing Enable code, directly or
  indirectly (via Chaco).

Example:

.. code-block:: python

  from traits.etsconfig.api import ETSConfig

  ETSConfig.toolkit = 'qt'
  from chaco.api import Plot
  from enable.api import ComponentEditor

  # ...

For the example, the application would use the Qt toolkit and the default kiva
backend [currently ``image``].

The format of the ``ETSConfig.toolkit`` property is
``<widget toolkit>[.<kiva backend>]``. The ``.<kiva backend>`` part is
*optional*. If no value is given for ``<kiva backend>``, the default kiva
backend is then selected by ``ESTConfig``. That means that if you want to use
the ``QPainter`` backend in your Qt-based application, you should do this:

.. code-block:: python

  from traits.etsconfig.api import ETSConfig
  ETSConfig.toolkit = 'qt.qpainter'

As another example, you might have a headless program (like a web server) which
wants to generate plots with Chaco. So you could use the ``null`` toolkit and
``image`` backend:

.. code-block:: python

  from traits.etsconfig.api import ETSConfig
  ETSConfig.toolkit = 'null.image'

``ETSConfig`` has a read-only property ``kiva_backend`` which you can read if
you'd like to know which kiva backend is currently selected, in case you haven't
explicitly selected a backend.


ETS_TOOLKIT
-----------

If you wish to test toolkit/backend combinations *without* the need for code
modifications, you can set the ``ETS_TOOLKIT`` environment variable before
running your application. ``ETSConfig`` will use the value of that environment
variable to initialize its ``toolkit`` and ``kiva_backend`` properties. Here's
what that looks like:

.. code-block:: shell

  $ export ETS_TOOLKIT=qt.qpainter
  $ python kiva/examples/kiva/kiva_explorer.py


enable.toolkits entrypoints
---------------------------

The allowed values for ``ETSConfig.toolkit`` are determined by the
``enable.toolkits`` `entrypoint <https://setuptools.readthedocs.io/en/latest/pkg_resources.html#entry-points>`_. An ``enable.toolkits`` entrypoint is a
:class:`~pyface.base_toolkit.Toolkit` object which when called with the name
of a class returns the appropriate version for the selected widget toolkit and
kiva backend.

The objects which are supplied by a toolkit/backend implementation are:

* :class:`CompiledPath`
* :class:`GraphicsContext`
* :class:`Window`
* :class:`NativeScrollBar`
* :func:`font_metrics_provider`

Because this is done via the setuptools entrypoint mechanism, it means that code
outside of Enable can contribute a custom toolkit and backends. To create a new
toolkit, you need to do the following:

1. Create a package for the toolkit
2. Add a ``toolkit.py`` module (its name is just a convention) which should
   contain a :class:`~pyface.base_toolkit.Toolkit` object initialized with the
   details of your package.
3. Add an ``enable.toolkits`` list to the ``entry_points`` keyword argument of
   your package's ``setup.py`` script's call to the ``setup()`` function. This
   list should contain a string with the format:
   ``<name> = my.package.toolkit:<Toolkit object name>``. This points to the
   ``Toolkit`` object created in the previous step.
4. For every kiva backend supported by your toolkit, add a ``<backend>.py``
   module to the package. This module must contain the objects listed above
   (``CompiledPath``, ``GraphicsContext``, etc.)

There's one more wrinkle to consider. :class:`~pyface.base_toolkit.Toolkit`
normally expects to be called with a ``<module>:<class>`` format string, but
Enable's usage of the ``Toolkit`` instance only passes a ``<class>`` name when
resolving objects. This is because ``<module>`` denotes the kiva backend, which
is not known until runtime. To provide this flexibility, the following wrapper
is used in Enable's built-in toolkits:

.. code-block:: python

  def _wrapper(func):
      def wrapped(name):
          # Prefix object lookups with the name of the configured kiva backend.
          return func(f"{ETSConfig.kiva_backend}:{name}")
      return wrapped

  toolkit = _wrapper(Toolkit("enable", "null", "enable.null"))
