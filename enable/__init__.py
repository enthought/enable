# (C) Copyright 2005-2023 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
""" A multi-platform object drawing library.
    Part of the Enable project of the Enthought Tool Suite.
"""
try:
    from enable._version import full_version as __version__
except ImportError:
    __version__ = "not-built"

__requires__ = [
    "numpy", "pillow", "traits>=6.2.0", "traitsui", "pyface>=7.2.0", "fonttools"
]

__extras_require__ = {
    # Dependencies for documentation
    "docs": ["enthought-sphinx-theme", "sphinx", "sphinx-copybutton"],
    # Dependencies for running enable/kiva's examples
    "examples": ["chaco", "mayavi", "scipy", "kiwisolver"],
    # Dependencies for constrained layout
    "layout": ["kiwisolver"],
    # Dependencies for PDF backend
    "pdf": ["reportlab"],
    # Dependencies for SVG backend
    "svg": ["pyparsing"],
    # Dependencies for Celiagg backend
    "celiagg": ["celiagg"],
    # Dependencies for Cairo backend
    "cairo": ["pycairo"],
    # Dependencies purely for running tests.
    "test": [
        "pyparsing",  # for enable.savage tests
        "PyPDF2<3.0",  # for pdf drawing tests in kiva.
        "setuptools",
    ],
    # Dependencies for PySide6
    "pyside6": ["pyface[pyside6]", "traitsui[pyside6]"],
    # Dependencies for PySide2
    "pyside2": ["pyface[pyside2]", "traitsui[pyside2]", "pillow<10"],
    # Dependencies for PyQt6
    "pyqt6": ["pyface[pyqt6]", "traitsui[pyqt6]"],
    # Dependencies for PyQt5
    "pyqt5": ["pyface[pyqt5]", "traitsui[pyqt5]", "pillow<10"],
    # Dependencies for WxPython
    "wx": ["pyface[wx]", "traitsui[wx]"],
    # Dependencies for null backend (nothing right now)
    "null": [],
}
