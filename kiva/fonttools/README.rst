fonttools
=========

This is a package provides a platform independent API for describing fonts
as well as locating and loading font information from the local system.

Only components exposed from ``kiva.font`` are public API. Other modules should
be considered as private implementation details.

Third-party libraries
---------------------

The ``afm`` and ``font_manager`` modules are based heavily on matplotlib's
``afm`` and ``font_manager`` modules. The latter has been modified to remove
dependencies on matplotlib. The original code is subject to matplotlib license.
Copies of licenses can be found in the LICENSES folder.
