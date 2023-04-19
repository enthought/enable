# (C) Copyright 2005-2022 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
from traitsui.api import toolkit

if toolkit().toolkit == "wx":
    from .wx.rgba_color_editor import RGBAColorEditor
elif toolkit().toolkit.startswith("qt"):
    from .qt4.rgba_color_editor import RGBAColorEditor
else:
    class RGBAColorEditor(object):
        """ An unimplemented toolkit object

        This is returned if an object isn't implemented by the selected
        toolkit.  It raises an exception if it is ever instantiated.
        """

        def __init__(self, *args, **kwargs):
            msg = "the %s backend doesn't implement RGBAColorEditor"
            raise NotImplementedError(msg % (toolkit().toolkit,))
