# (C) Copyright 2005-2022 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
""" Trait definition for a wxPython-based Kiva font.
"""

from enable.trait_defs.kiva_font_trait import KivaFont as _KivaFont

# old KivaFont defaulted to "modern" family rather than "default"
KivaFont = _KivaFont("modern 12")
