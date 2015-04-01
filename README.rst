=========================================
enable: low-level drawing and interaction
=========================================

http://github.enthought.com/enable

.. image:: https://travis-ci.org/enthought/enable.svg?branch=master
   :target: https://travis-ci.org/enthought/enable
   :alt: Build status

.. image:: https://coveralls.io/repos/enthought/enable/badge.png
   :target: https://coveralls.io/r/enthought/enable
   :alt: Coverage status

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
the Enable project:

* `Setuptools <https://pypi.python.org/pypi/setuptools>`_
* `Numpy <http://pypi.python.org/pypi/numpy>`_
* `SWIG <http://www.swig.org/>`_
* (on Linux) X11-devel (development tools for X11)
* (on Mac OS X) `Cython <http://www.cython.org>`_

Enable/Kiva also have the following requirements:

.. rubric:: Runtime:

* `Numpy <http://pypi.python.org/pypi/numpy>`_
* `PIL <http://www.pythonware.com/products/pil>`._
* `traits <https://pypi.python.org/pypi/traits>`_
* `traitsui <https://pypi.python.org/pypi/traitsui>`_
* `pyface <https://pypi.python.org/pypi/pyface>`_
* `apptools 4.3.0 <https://pypi.python.org/pypi/apptools/>`_
* `kiwisolver <https://pypi.python.org/pypi/kiwisolver>`_

.. rubric:: Optional:

* (GL backend) `pyglet == 1.1.4 <https://bitbucket.org/pyglet/pyglet/get/pyglet-1.1.4.zip>`_
* (GL backend) `pygarrayimage <https://pypi.python.org/pypi/pygarrayimage>`_
* (SVG backend) `PyParsing <https://pypi.python.org/pypi/pyparsing>`_
* (PDF backend) `ReportLab Toolkit <= 3.1 <http://www.reportlab.org/rl_toolkit.html/>`_
  backend support in Kiva.

