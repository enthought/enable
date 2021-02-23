# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
"""
Kiva Explorer
=============

Interactive editor for exploring Kiva drawing commands.

"""
import time

from traits.api import Any, Code, Float, Instance, Property, Str
from traitsui.api import HSplit, ModelView, UItem, VGroup, View

from enable.api import Component, ComponentEditor


default_script = """# Write your code here.
# The graphics context is available as gc.

from math import pi
from kiva.api import CAP_ROUND, JOIN_ROUND, Font

with gc:
    gc.set_fill_color((1.0, 1.0, 0.0, 1.0))
    gc.arc(200, 200, 100, 0, 2*pi)
    gc.fill_path()

    with gc:
        gc.set_font(Font('Times New Roman', size=24))
        gc.translate_ctm(200, 200)
        for i in range(0, 12):
            gc.set_fill_color((i/12.0, 0.0, 1.0-(i/12.0), 0.75))
            gc.rotate_ctm(2*pi/12.0)
            gc.show_text_at_point("Hello World", 20, 0)

    gc.set_stroke_color((0.0, 0.0, 1.0, 1.0))
    gc.set_line_width(7)
    gc.set_line_join(JOIN_ROUND)
    gc.set_line_cap(CAP_ROUND)
    gc.rect(100, 400, 50, 50)
    gc.stroke_path()

"""


class ScriptedComponent(Component):
    """ An Enable component that draws its mainlayer from a script
    """

    #: kiva drawing code for mainlayer
    draw_script = Code(default_script)

    #: any errors which occur
    error = Str

    #: how long did the last draw take
    last_draw_time = Float(0.0)
    fps_display = Property(Str, depends_on="last_draw_time")

    #: compiled code
    _draw_code = Any

    def _draw_mainlayer(self, gc, view_bounds=None, mode="default"):
        """ Try running the compiled code with the graphics context as `gc`
        """
        with gc:
            try:
                self.error = ""
                start_time = time.time()
                exec(self._draw_code, {}, {"gc": gc})
                self.last_draw_time = time.time() - start_time
            except Exception as exc:
                self.error = str(exc)

    def _compile_script(self):
        """ Try compiling the script to bytecode
        """
        try:
            self.error = ""
            return compile(self.draw_script, "<script>", "exec")
        except SyntaxError as exc:
            self.error = str(exc)
            return None

    def _draw_script_changed(self):
        code = self._compile_script()
        if code is not None:
            self._draw_code = code
            self.request_redraw()

    def __draw_code_default(self):
        code = self._compile_script()
        if code is None:
            code = compile("", "<script>", "exec")
        return code

    def _get_fps_display(self):
        if self.last_draw_time == 0.0:
            return ""

        draw_time_ms = self.last_draw_time * 1000.0
        draw_fps = 1000.0 / draw_time_ms
        return "{:.2f}ms ({:.2f} fps)".format(draw_time_ms, draw_fps)


class ScriptedComponentView(ModelView):
    """ ModelView of a ScriptedComponent displaying the script and image
    """

    #: the component we are editing
    model = Instance(ScriptedComponent, ())

    view = View(
        HSplit(
            VGroup(
                UItem("model.draw_script"),
                UItem(
                    "model.error",
                    visible_when="model.error != ''",
                    style="readonly",
                    height=100,
                ),
            ),
            VGroup(
                UItem("model", editor=ComponentEditor(), springy=True),
                UItem("model.fps_display", style="readonly"),
            ),
        ),
        resizable=True,
        title="Kiva Explorer",
    )


# "popup" is a magically named variable for the etsdemo application which will
# cause this demo to be run as a popup rather than trying to compress it into
# a tab on the application
popup = ScriptedComponentView()

if __name__ == "__main__":
    popup.configure_traits()
