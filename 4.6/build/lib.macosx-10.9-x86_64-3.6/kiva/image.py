# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
""" This backend supports Kiva drawing into memory buffers.

    Though this can be used to perform drawing in non-GUI applications,
    the Image backend is also used by many of the GUI backends to draw into
    the memory space of the graphics contexts.
"""

# Soon the Agg subpackage will be renamed for real.  For now, just
# proxy the imports.
from .agg import GraphicsContextArray as GraphicsContext

# FontType will be unified with the Kiva FontType soon.
from .agg import AggFontType as FontType

# GraphicsContextSystem wraps up platform- and toolkit- specific low
# level calls with a GraphicsContextArray.  Eventually this low-level code
# will be moved out of the Agg subpackage, and we won't have be importing
# this here.
from .agg import GraphicsContextSystem, Image

# CompiledPath is an object that can efficiently store paths to be reused
# multiple times.
from .agg import CompiledPath


def font_metrics_provider():
    """ Create an object to be used for querying font metrics.
    """

    return GraphicsContext((1, 1))
