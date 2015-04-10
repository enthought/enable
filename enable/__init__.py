#  Copyright (c) 2007-2014 by Enthought, Inc.
#  All rights reserved.
""" A multi-platform object drawing library.
    Part of the Enable project of the Enthought Tool Suite.
"""
import sys
from ._version import full_version as __version__

__all__ = [
    '__version__',
]

__requires__ = [
    'numpy',
    'traits',
    'traitsui',
    'PIL',
    'pyface',
]

# Cython is only necessary to build the quartz backend.
if sys.platform == 'darwin':
    __requires__.append('cython')
