#!/usr/bin/env python
# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
""" Compare the output of various Kiva SVG implementations against other
renderers.
"""

from io import StringIO
import glob
import logging
import os
import pstats
import sys
from xml.etree import cElementTree as ET
import warnings

from PIL import Image
import numpy as np

from enable.api import Component
from enable.component_editor import ComponentEditor
from traits.api import (
    Any, Button, Dict, HasTraits, HTML, Instance, List, Property, Str,
    on_trait_change
)
from traitsui.api import (
    EnumEditor, HGroup, HSplit, Item, Tabbed, VGroup, View, VSplit
)

from enable.savage.svg import document
from enable.savage.trait_defs.ui.svg_editor import SVGEditor
from enable.savage.svg.backends.wx.renderer import Renderer as WxRenderer
from enable.savage.svg.backends.kiva.renderer import Renderer as KivaRenderer

from .crosshair import Crosshair, MultiController
from .profile_this import ProfileThis
from .sike import Sike
from .svg_component import ImageComponent, SVGComponent
from .xml_view import xml_to_tree, xml_tree_editor


logger = logging.getLogger()
this_dir = os.path.abspath(os.path.dirname(__file__))
SVG_TS_NS = "http://www.w3.org/2000/02/svg/testsuite/description/"


class ComponentTrait(Instance):
    """ Convenience trait for Enable Components.
    """

    def __init__(self, **kwds):
        kwds.setdefault("klass", Component)
        super(ComponentTrait, self).__init__(**kwds)

    def create_editor(self):
        return ComponentEditor()


def normalize_text(text):
    """ Utility to normalize the whitespace in text.

    This is used in order to prevent wx's HTML widget from overzealously trying
    to interpret the whitespace as indicating preformatted text.
    """
    return " ".join(text.strip().split())


def activate_tool(component, tool):
    """ Add and activate an overlay tool.
    """
    component.tools.append(tool)
    component.overlays.append(tool)
    component.active_tool = tool
    return tool


class Comparator(HasTraits):
    """ The main application.
    """

    # Configuration traits ##################################################

    # The root directory of the test suite.
    suitedir = Str()

    # Mapping of SVG basenames to their reference PNGs. Use None if there is no
    # reference PNG.
    svg_png = Dict()

    # The list of SVG file names.
    svg_files = List()

    # The name of the default PNG file to display when no reference PNG exists.
    default_png = Str(os.path.join(this_dir, "images/default.png"))

    # State traits ##########################################################

    # The currently selected SVG file.
    current_file = Str()
    abs_current_file = Property(depends_on=["current_file"])

    # The current XML ElementTree root Element and its XMLTree view model.
    current_xml = Any()
    current_xml_view = Any()

    # The profilers.
    profile_this = Instance(ProfileThis, args=())

    # GUI traits ############################################################

    # The text showing the current mouse coordinates over any of the
    # components.
    mouse_coords = Property(Str, depends_on=["ch_controller.svg_coords"])

    # Move forward and backward through the list of SVG files.
    move_forward = Button(">>")
    move_backward = Button("<<")

    # The description of the test.
    description = HTML()

    document = Instance(document.SVGDocument)

    # The components to view.
    kiva_component = ComponentTrait(klass=SVGComponent)
    ref_component = ComponentTrait(klass=ImageComponent, args=())
    ch_controller = Instance(MultiController)

    # The profiler views.
    parsing_sike = Instance(Sike, args=())
    drawing_sike = Instance(Sike, args=())
    wx_doc_sike = Instance(Sike, args=())
    kiva_doc_sike = Instance(Sike, args=())

    traits_view = View(
        Tabbed(
            VGroup(
                HGroup(
                    Item(
                        "current_file",
                        editor=EnumEditor(name="svg_files"),
                        style="simple",
                        width=1.0,
                        show_label=False,
                    ),
                    Item(
                        "move_backward",
                        show_label=False,
                        enabled_when="svg_files.index(current_file) != 0",
                    ),
                    Item(
                        "move_forward",
                        show_label=False,
                        enabled_when=("svg_files.index(current_file) "
                                      "!= len(svg_files)-1"),
                    ),
                ),
                VSplit(
                    HSplit(
                        Item(
                            "description",
                            label="Description",
                            show_label=False,
                        ),
                        Item(
                            "current_xml_view",
                            editor=xml_tree_editor,
                            show_label=False,
                        ),
                    ),
                    HSplit(
                        Item(
                            "document", editor=SVGEditor(), show_label=False
                        ),
                        Item("kiva_component", show_label=False),
                        Item("ref_component", show_label=False),
                        # TODO: Item('agg_component', show_label=False),
                    ),
                ),
                label="SVG",
            ),
            Item(
                "parsing_sike",
                style="custom",
                show_label=False,
                label="Parsing Profile",
            ),
            Item(
                "drawing_sike",
                style="custom",
                show_label=False,
                label="Kiva Drawing Profile",
            ),
            Item(
                "wx_doc_sike",
                style="custom",
                show_label=False,
                label="Creating WX document",
            ),
            Item(
                "kiva_doc_sike",
                style="custom",
                show_label=False,
                label="Creating WX document",
            ),
        ),
        width=1280,
        height=768,
        resizable=True,
        statusbar="mouse_coords",
        title="SVG Comparator",
    )

    def __init__(self, **traits):
        super(Comparator, self).__init__(**traits)
        kiva_ch = activate_tool(
            self.kiva_component, Crosshair(self.kiva_component)
        )
        ref_ch = activate_tool(
            self.ref_component, Crosshair(self.ref_component)
        )
        self.ch_controller = MultiController(kiva_ch, ref_ch)

    @classmethod
    def fromsuitedir(cls, dirname, **traits):
        """ Find all SVG files and their related reference PNG files under
        a directory.

        This assumes that the SVGs are located under <dirname>/svg/ and the
        related PNGs under <dirname>/png/ and that there are no subdirectories.
        """
        dirname = os.path.abspath(dirname)
        svgs = glob.glob(os.path.join(dirname, "svg", "*.svg"))
        pngdir = os.path.join(dirname, "png")
        d = {}
        for svg in svgs:
            png = None
            base = os.path.splitext(os.path.basename(svg))[0]
            for prefix in ("full-", "basic-", "tiny-", ""):
                fn = os.path.join(pngdir, prefix + base + ".png")
                if os.path.exists(fn):
                    png = os.path.basename(fn)
                    break
            d[os.path.basename(svg)] = png
        svgs = sorted(d)
        x = cls(suitedir=dirname, svg_png=d, svg_files=svgs, **traits)
        x.current_file = svgs[0]
        return x

    def display_reference_png(self, filename):
        """ Read the image file and shove its data into the display component.
        """
        img = Image.open(filename)
        arr = np.array(img)
        self.ref_component.image = arr

    def display_test_description(self):
        """ Extract the test description for display.
        """
        html = ET.Element("html")

        title = self.current_xml.find(".//{http://www.w3.org/2000/svg}title")
        if title is not None:
            title_text = title.text
        else:
            title_text = os.path.splitext(self.current_file)[0]
        p = ET.SubElement(html, "p")
        b = ET.SubElement(p, "b")
        b.text = "Title: "
        b.tail = title_text

        desc_text = None
        version_text = None
        desc = self.current_xml.find(".//{http://www.w3.org/2000/svg}desc")
        if desc is not None:
            desc_text = desc.text
        else:
            testcase = self.current_xml.find(f".//{{SVG_TS_NS}}SVGTestCase")
            if testcase is not None:
                desc_text = testcase.get("desc", None)
                version_text = testcase.get("version", None)
        if desc_text is not None:
            p = ET.SubElement(html, "p")
            b = ET.SubElement(p, "b")
            b.text = "Description: "
            b.tail = normalize_text(desc_text)

        if version_text is None:
            script = self.current_xml.find(f".//{{SVG_TS_NS}}OperatorScript")
            if script is not None:
                version_text = script.get("version", None)
        if version_text is not None:
            p = ET.SubElement(html, "p")
            b = ET.SubElement(p, "b")
            b.text = "Version: "
            b.tail = version_text

        paras = self.current_xml.findall(f".//{{SVG_TS_NS}}Paragraph")
        if len(paras) > 0:
            div = ET.SubElement(html, "div")
            for para in paras:
                p = ET.SubElement(div, "p")
                p.text = normalize_text(para.text)
                # Copy over any children elements like <a>.
                p[:] = para[:]

        tree = ET.ElementTree(html)
        f = StringIO()
        tree.write(f)
        text = f.getvalue()
        self.description = text

    def locate_file(self, name, kind):
        """ Find the location of the given file in the suite.

        Parameters
        ----------
        name : str
            Path of the file relative to the suitedir.
        kind : either 'svg' or 'png'
            The kind of file.

        Returns
        -------
        path : str
            The full path to the file.
        """
        return os.path.join(self.suitedir, kind, name)

    def _kiva_component_default(self):
        return SVGComponent(profile_this=self.profile_this)

    def _move_backward_fired(self):
        idx = self.svg_files.index(self.current_file)
        idx = max(idx - 1, 0)
        self.current_file = self.svg_files[idx]

    def _move_forward_fired(self):
        idx = self.svg_files.index(self.current_file)
        idx = min(idx + 1, len(self.svg_files) - 1)
        self.current_file = self.svg_files[idx]

    def _get_abs_current_file(self):
        return self.locate_file(self.current_file, "svg")

    def _current_file_changed(self, new):
        # Reset the warnings filters. While it's good to only get 1 warning per
        # file, we want to get the same warning again if a new file issues it.
        warnings.resetwarnings()

        self.profile_this.start("Parsing")
        self.current_xml = ET.parse(self.abs_current_file).getroot()
        self.current_xml_view = xml_to_tree(self.current_xml)
        resources = document.ResourceGetter.fromfilename(self.abs_current_file)
        self.profile_this.stop()
        try:
            self.profile_this.start("Creating WX document")
            self.document = document.SVGDocument(
                self.current_xml, resources=resources, renderer=WxRenderer
            )
        except Exception:
            logger.exception("Error parsing document %s", new)
            self.document = None

        self.profile_this.stop()

        try:
            self.profile_this.start("Creating Kiva document")
            self.kiva_component.document = document.SVGDocument(
                self.current_xml, resources=resources, renderer=KivaRenderer
            )
        except Exception:
            logger.exception("Error parsing document %s", new)
            self.kiva_component.document

        self.profile_this.stop()

        png_file = self.svg_png.get(new, None)
        if png_file is None:
            png_file = self.default_png
        else:
            png_file = self.locate_file(png_file, "png")
        self.display_test_description()
        self.display_reference_png(png_file)

    def _get_mouse_coords(self):
        if self.ch_controller is None:
            return ""
        else:
            return "%1.3g %1.3g" % self.ch_controller.svg_coords

    @on_trait_change("profile_this:profile_ended")
    def _update_profiling(self, new):
        if new is not None:
            name, p = new
            stats = pstats.Stats(p)
            if name == "Parsing":
                self.parsing_sike.stats = stats
            elif name == "Drawing":
                self.drawing_sike.stats = stats
            elif name == "Creating WX document":
                self.wx_doc_sike.stats = stats
            elif name == "Creating Kiva document":
                self.kiva_doc_sike.stats = stats


class OpenClipartComparator(Comparator):
    """ Locate SVG files and PNGs in directories laid out like the OpenClipart
    packages.
    """

    @classmethod
    def fromsuitedir(cls, dirname, **traits):
        """ Load SVG and reference PNGs from an OpenClipart directory.
        """
        dirname = os.path.abspath(dirname)

        def remove_prefix(path, dirname=dirname):
            if path.startswith(dirname + os.path.sep):
                path = path[len(dirname) + 1:]
            return path

        svg_png = {}
        for d, dirs, files in os.walk(dirname):
            for fn in files:
                fn = os.path.join(d, fn)
                base, ext = os.path.splitext(fn)
                if ext == ".svg":
                    png = os.path.join(d, base + ".png")
                    if os.path.exists(png):
                        png = remove_prefix(png)
                    else:
                        png = None
                    svg = remove_prefix(fn)
                    svg_png[svg] = png

        svgs = sorted(svg_png)
        x = cls(suitedir=dirname, svg_png=svg_png, svg_files=svgs, **traits)
        x.current_file = svgs[0]
        return x

    def locate_file(self, name, kind):
        return os.path.join(self.suitedir, name)


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--openclipart",
        action="store_true",
        help=("The suite is in OpenClipart layout rather than the SVG test "
              "suite layout."),
    )
    parser.add_argument(
        "--suitedir",
        nargs="?",
        default=os.path.join(this_dir, "w3c_svg_11"),
        help="The directory with the test suite. [default: %(default)s]",
    )

    args = parser.parse_args()
    logging.basicConfig(stream=sys.stdout)
    if args.openclipart:
        klass = OpenClipartComparator
    else:
        klass = Comparator
    if os.path.isfile(args.suitedir):
        # We were given a single SVG file.
        if args.openclipart:
            suitedir, svg = os.path.split(args.suitedir)
        else:
            svgdir, svg = os.path.split(args.suitedir)
            suitedir = os.path.split(svgdir)[0]
        c = klass(suitedir=suitedir)
        png = os.path.splitext(svg)[0] + ".png"
        if not os.path.exists(c.locate_file(png, "png")):
            png = None
        c.svg_png = {svg: png}
        c.svg_files = [svg]
        c.current_file = svg
    else:
        c = klass.fromsuitedir(args.suitedir)
    c.configure_traits()


if __name__ == "__main__":
    main()
