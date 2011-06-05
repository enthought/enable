#------------------------------------------------------------------------------
# Copyright (c) 2011, Enthought, Inc.
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in enthought/LICENSE.txt and may be redistributed only
# under the conditions described in the aforementioned license.  The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
# Thanks for using Enthought open source!
#------------------------------------------------------------------------------

from kiva.ps import CompiledPath, PSGC as GraphicsContext

class NativeScrollBar(object):
    pass

class Window(object):
    pass

def font_metrics_provider():
    from kiva.fonttools import Font
    gc = GraphicsContext((1, 1))
    gc.set_font(Font())
    return gc
