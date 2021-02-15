# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
from traits.etsconfig.api import ETSConfig

if ETSConfig.toolkit == "wx":
    from .wx.rgba_color_editor import RGBAColorEditor
elif ETSConfig.toolkit.startswith("qt"):
    from .qt4.rgba_color_editor import RGBAColorEditor
else:
    RGBAColorEditor = None
