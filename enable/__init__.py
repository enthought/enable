#  Copyright (c) 2007-2014 by Enthought, Inc.
#  All rights reserved.
""" A multi-platform object drawing library.
    Part of the Enable project of the Enthought Tool Suite.
"""
import sys

from kiva._version import full_version as __version__

__requires__ = [
    'traitsui',
    'PIL',
    'kiwisolver',
]

# Cython is only necessary to built the quartz backend.
if sys.platform == 'darwin':
    __requires__.append('cython')
