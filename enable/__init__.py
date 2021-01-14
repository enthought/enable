#  Copyright (c) 2007-2014 by Enthought, Inc.
#  All rights reserved.
""" A multi-platform object drawing library.
    Part of the Enable project of the Enthought Tool Suite.
"""
from ._version import full_version as __version__

__requires__ = [
    'numpy',
    'traits',
    'traitsui',
    'pyface',
    'fonttools'
]

# Do not force installation of pillow if PIL is already available.
try:
    import PIL
except ImportError:
    __requires__.append('pillow')

__extras_require__ = {
    # Dependencies for running enable/kiva's examples
    'examples': [
        'chaco',
        'mayavi',
        'scipy',
        'kiwisolver',
        'pyglet'
    ],
    # Dependencies for GL backend support
    'gl': [
        'pygarrayimage',
        'pyglet',
    ],
    # Dependencies for constrained layout
    'layout': [
        'kiwisolver',
    ],
    # Dependencies for PDF backend
    'pdf': [
        'reportlab',
    ],
    # Dependencies for SVG backend
    'svg': [
        'pyparsing',
    ],
    # Dependencies purely for running tests.
    'test': [
        'hypothesis',
        'PyPDF2',    # for pdf drawing tests in kiva.
    ]
}
