# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
from kiva.celiagg import CompiledPath, GraphicsContext  # noqa


class NativeScrollBar(object):
    pass


class Window(object):
    pass


def font_metrics_provider():
    from kiva.api import Font

    gc = GraphicsContext((1, 1))
    gc.set_font(Font())
    return gc
