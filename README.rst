=========================================
enable: low-level drawing and interaction
=========================================

http://github.enthought.com/enable


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

You must have the following libraries installed before building or installing
the Enable project:

* `distribute <http://pypi.python.org/pypi/distribute>`_
* `SWIG <http://www.swig.org/>`_
* `Cython <http://www.cython.org>`_
* `Numpy <http://pypi.python.org/pypi/numpy>`_
* `ReportLab Toolkit <http://www.reportlab.org/rl_toolkit.html/>`_ for PDF
  backend support in Kiva.
* (on Linux) X11-devel (development tools for X11)
