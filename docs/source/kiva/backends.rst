Kiva Backends
=============

GUI-capable
-----------
Each of these backends can be used to draw the contents of windows in a
graphical user interface.

kiva.agg/image
~~~~~~~~~~~~~~
This is a wrapper of the popular Anti-Grain Geometry C++ library. It is the
current default backend.

cairo
~~~~~
A backend based on the `Cairo graphics library <https://www.cairographics.org/>`_.

celiagg
~~~~~~~
A newer wrapper of Anti-Grain Geometry which is maintained outside of
kiva/enable.

gl
~~
OpenGL drawing. This backend is quite limited compared to others.

qpainter
~~~~~~~~
Qt ``QPainter`` drawing. This is only availble with the Qt toolkit.

quartz
~~~~~~
macOS Quartz graphics (ie `CGContext <https://developer.apple.com/documentation/coregraphics/cgcontext>`_).
This is only available on macOS.

File-only
---------
Each of these backends can be used to create an output file.

pdf
~~~
A backend which writes PDF files.

ps
~~
A backend which writes PostScript files.

svg
~~~
A backend which writes SVG files.
