# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
""" An Enable component to render SVG documents.
"""

import sys
import time

from enable.api import Component
from traits.api import Any, Array, Bool, Float
from kiva.api import Font


if sys.platform == "win32":
    now = time.clock
else:
    now = time.time


class SVGComponent(Component):
    """ An Enable component to render SVG documents.
    """

    # The SVGDocument.
    document = Any()

    # The number of seconds it took to do the last draw.
    last_render = Float()

    # The profile manager.
    profile_this = Any()
    should_profile = Bool(False)

    def _draw_mainlayer(self, gc, view_bounds=None, mode="default"):
        if self.should_profile and self.profile_this is not None:
            # Only profile the first draw.
            self.should_profile = False
            self.profile_this.start("Drawing")
        start = now()
        gc.clear()

        width, height = self.bounds
        gc.save_state()
        if self.document is None:
            # fixme: The Mac backend doesn't accept style/width as non-integers
            #        in set_font, but does for select_font...
            if sys.platform == "darwin":
                gc.select_font("Helvetica", 36)
            else:
                gc.set_font(Font("Helvetica", 36))
            gc.show_text_at_point("Could not parse document.", 20, height - 56)
            gc.restore_state()
            if self.profile_this is not None:
                self.profile_this.stop()
            return

        try:
            # SVG origin is upper right with y positive is down.
            # Set up the transforms to fix this up.
            # FIXME: if the rendering stage fails, all subsequent renders are
            # vertically flipped
            gc.translate_ctm(0, height)
            # TODO: bother with zoom?
            # TODO: inspect the view bounds and scale to the shape of the
            # component?
            scale = 1.0
            gc.scale_ctm(scale, -scale)
            self.document.render(gc)
            self.last_render = now() - start

        finally:
            gc.restore_state()
            if self.profile_this is not None:
                self.profile_this.stop()

    def _document_changed(self):
        self.should_profile = True
        self.invalidate_and_redraw()


class ImageComponent(Component):
    """ Simple component that just renders an RGB(A) array in the upper left
    hand corner of the component.
    """

    # The RGB(A) data.
    image = Array()

    def _draw_mainlayer(self, gc, view_bounds=None, mode="default"):
        gc.clear()
        if len(self.image.shape) != 3:
            # No image.
            return
        gc.save_state()
        try:
            width, height = self.bounds
            img_height, img_width = self.image.shape[:2]
            gc.draw_image(
                self.image, (0.0, height - img_height, img_width, img_height)
            )
        finally:
            gc.restore_state()

    def _image_changed(self):
        self.invalidate_and_redraw()
