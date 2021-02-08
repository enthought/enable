=========================================
enable: low-level drawing and interaction
=========================================

https://docs.enthought.com/enable

.. image:: https://github.com/enthought/enable/workflows/Test%20with%20EDM/badge.svg
   :target: https://github.com/enthought/enable/actions?query=workflow%3A%22Test+with+EDM%22
   :alt: Build status

The Enable *project* provides two related multi-platform *packages* for drawing
GUI objects.

- **Enable**: An object drawing library that supports containment and event
  notification.
- **Kiva**: A multi-platform DisplayPDF vector drawing engine.

Enable
------

The Enable package is a multi-platform object drawing library built on top of
Kiva. The core of Enable is a container/component model for drawing and event
notification. The core concepts of Enable are:

- Component
- Container
- Events (mouse, drag, and key events)

Enable provides a high-level interface for creating GUI objects, while
enabling a high level of control over user interaction. Enable is a supporting
technology for the Chaco and BlockCanvas projects.


Kiva
----

Kiva is a multi-platform DisplayPDF vector drawing engine that supports
multiple output backends, including Windows, GTK, and Macintosh native
windowing systems, a variety of raster image formats, PDF, and Postscript.

DisplayPDF is more of a convention than an actual specification. It is a
path-based drawing API based on a subset of the Adobe PDF specification.
Besides basic vector drawing concepts such as paths, rects, line sytles, and
the graphics state stack, it also supports pattern fills, antialiasing, and
transparency. Perhaps the most popular implementation of DisplayPDF is
Apple's Quartz 2-D graphics API in Mac OS X.

Kiva Features
`````````````
Kiva currently implements the following features:

- paths and compiled paths; arcs, bezier curves, rectangles
- graphics state stack
- clip stack, disjoint rectangular clip regions
- raster image blitting
- arbitrary affine transforms of the graphics context
- bevelled and mitered joins
- line width, line dash
- Freetype or native fonts
- RGB, RGBA, or grayscale color depths
- transparency

Prerequisites
-------------

You must have the following libraries installed before building
the Enable/Kiva project:

- `Setuptools <https://pypi.org/project/setuptools>`_
- `Numpy <https://pypi.org/project/numpy/>`_
- `SWIG <http://www.swig.org/>`_
- `fonttools <https://pypi.org/project/FontTools>`_
- (on Linux) X11-devel (development tools for X11)
- (on Linux) libglu1-mesa-dev (OpenGL utility library for development)
- (on Mac OS X) `Cython <https://cython.org/>`_

Enable/Kiva also have the following requirements:

.. rubric:: Runtime:

- `Numpy <https://pypi.org/project/numpy/>`_
- `PIL <https://www.pythonware.com/products/pil/>`_ or preferably `pillow <https://pypi.org/project/Pillow/>`_
- `traits <https://pypi.org/project/traits>`_
- `traitsui <https://pypi.org/project/traitsui>`_
- `pyface <https://pypi.org/project/pyface>`_

.. rubric:: Optional:

- `apptools <https://pypi.org/project/apptools/>`_
- (Qt backend) `PySide <https://pypi.org/project/PySide>`_ or `PyQt4 <https://pypi.org/project/PyQt4>`_
- (WX backend) `WxPython <https://pypi.org/project/wxPython/>`_
- (GL backend) `pyglet <https://pypi.org/project/pyglet/>`_
- (GL backend) `pygarrayimage <https://pypi.org/project/pygarrayimage>`_
- (SVG backend) `PyParsing <https://pypi.org/project/pyparsing>`_
- (PDF backend) `ReportLab Toolkit <https://www.reportlab.com/dev/install/version_3_and_up/>`_
- (Cairo backend) `PyCairo <https://cairographics.org/releases/>`_
- (Constrained layout) `kiwisolver <https://pypi.org/project/kiwisolver>`_
