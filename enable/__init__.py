# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
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
from ._version import full_version as __version__

__requires__ = [
    "numpy", "pillow", "traits", "traitsui", "pyface", "fonttools"
]

__extras_require__ = {
    # Dependencies for running enable/kiva's examples
    "examples": ["chaco", "mayavi", "scipy", "kiwisolver", "pyglet"],
    # Dependencies for GL backend support
    "gl": ["pygarrayimage", "pyglet"],
    # Dependencies for constrained layout
    "layout": ["kiwisolver"],
    # Dependencies for PDF backend
    "pdf": ["reportlab"],
    # Dependencies for SVG backend
    "svg": ["pyparsing"],
    # Dependencies purely for running tests.
    "test": [
        "hypothesis",
        "PyPDF2",  # for pdf drawing tests in kiva.
        "setuptools",
    ],
}
