# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
""" Initialize this backend.
"""
from pyface.base_toolkit import Toolkit
from traits.etsconfig.api import ETSConfig


def _wrapper(func):
    def wrapped(name):
        # Prefix object lookups with the name of the configured kiva backend.
        return func(f'{ETSConfig.kiva_backend}:{name}')
    return wrapped


toolkit = _wrapper(Toolkit("enable", "wx", "enable.wx"))
