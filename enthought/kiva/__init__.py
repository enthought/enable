"""
A multi-platform DisplayPDF vector drawing engine. 
Part of the Enable project of the Enthought Tool Suite.

This file pulls in the Kiva constants and does the detection of what backend
to use via the KIVA_WISHLIST environment variable.
"""

#------------------------------------------------------------------------------
# Copyright (c) 2005, Enthought, Inc.
# some parts copyright 2002 by Space Telescope Science Institute
# All rights reserved.
# 
# This software is provided without warranty under the terms of the BSD
# license included in enthought/LICENSE.txt and may be redistributed only
# under the conditions described in the aforementioned license.  The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
# Thanks for using Enthought open source!
#------------------------------------------------------------------------------

import os

from enthought.etsconfig.api import ETSConfig

from constants import *
from fonttools import Font   # relative import; not the fonttools project!

# The code is borrowed almost exactly from the AnyGui __init__.py
# code.  Nice code.    

# This module is really just a shell around the various core2d backends.  It 
# loops through a list of possible backends until it finds on that is 
# available.  You can override the standard list of backends by setting the 
# environment variable KIVA_WISHLIST before importing.  The environment
# variable is a space separated list of backends.  The first available
# one in the list is chosen.


_kiva_only_backends = 'gl image pdf svg ps cairo'
_backends = 'wx qt4 ' + _kiva_only_backends

_backend = None
DEBUG = 1
wishlist = os.environ.get('KIVA_WISHLIST', _backends).split()

# Symbols we import from the selected backend
backend_symbols = ('GraphicsContext', 'Canvas', 'CompiledPath',
    'font_metrics_provider')

def _dotted_import(name):
    # version of __import__ which handles dotted names
    # copied from python docs for __import__
    import string
    mod = __import__(name, globals(), locals(), [])
    components = string.split(name, '.')
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def _try_backend(name):
    if name == 'null':
        # If we are given a 'null' UI toolkit, just use the image backend.
        name = 'image'
    try:
        mod = _dotted_import('backend_%s' % name)
        for key in backend_symbols:
            globals()[key] = mod.__dict__[key]
    except (ImportError, AttributeError, KeyError):
        if DEBUG and not (str(DEBUG) in _backends and not DEBUG==name):
            import traceback
            traceback.print_exc()
        return False
    else:
        global _backend
        _backend = name
        return True


def _backend_passthrough():
    global _backends

    # Remove from the list all known backends that are in the wishlist.
    _backends = _backends.split()
    _backends = [b for b in _backends if not b in wishlist]

    if wishlist:
        # Replace a '*' in the wishlist with any known backends that aren't
        # already there.
        try:
            idx = wishlist.index('*')
            wishlist[idx:idx+1] = _backends
        except ValueError: pass
        _backends = wishlist

    for name in _backends:
        if _try_backend(name):
            return

    raise RuntimeError, "no usable backend found"


def backend():
    'Return the name of the current backend'
    if not _backend:
        raise RuntimeError('no backend exists')
    return _backend


# See if the toolkit has already been selected and the user hasn't explicitly
# set their KIVA_WISHLIST environment variable
if ETSConfig.toolkit:
    if "KIVA_WISHLIST" in os.environ:
        _backend_passthrough()
    elif not _try_backend(ETSConfig.toolkit):
        raise RuntimeError("no kiva backend for %s" % ETSConfig.toolkit)
else:
    _backend_passthrough()

    # Impose the kiva selection unless it is a kiva only backend.
    if _backend not in _kiva_only_backends.split():
        ETSConfig.toolkit = _backend
