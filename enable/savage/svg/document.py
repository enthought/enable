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
    SVGDocument
"""
from io import BytesIO
import warnings
import math
from functools import wraps
import os

import urllib.request
import urllib.parse as urlparse
from xml.etree import cElementTree as ET
from xml.etree.cElementTree import ParseError


import numpy

from . import css
from .css.colour import colourValue
from .css import values
from .attributes import paintValue
from .svg_regex import svg_parser

from enable.savage.svg.backends.null.null_renderer import (
    NullRenderer,
    AbstractGradientBrush,
)


class XMLNS(object):
    """ Utility object for dealing the namespaced names quoted the way
    ElementTree requires.
    """

    def __init__(self, url):
        self.__url = url

    def __getattr__(self, attr):
        return self[attr]

    def __getitem__(self, key):
        return "{%s}%s" % (self.__url, key)


XLink = XMLNS("http://www.w3.org/1999/xlink")
XML = XMLNS("http://www.w3.org/XML/1998/namespace")
SVG = XMLNS("http://www.w3.org/2000/svg")


def normalize_href(href):
    """ Normalize an href to remove url(...) and xpointer(id(...)) extraneous
    bits.

    Parameters
    ----------
    href : str

    Returns
    -------
    uri : str
        A URI (or maybe a file system path) to the resource.
    fragment : str
        The normalized #fragment, if any
    """
    if href.startswith("url(") and href.endswith(")"):
        href = href[4:-1]
    scheme, netloc, path, params, query, fragment = urlparse.urlparse(href)
    # Normalize xpointer(id(...)) references.
    if fragment.startswith("xpointer(id(") and fragment.endswith("))"):
        # FIXME: whitespace?
        fragment = fragment[12:-2]
    uri = urlparse.urlunparse((scheme, netloc, path, params, query, ""))
    return uri, fragment


def attrAsFloat(node, attr, defaultValue="0"):
    val = node.get(attr, defaultValue)
    # TODO: process stuff like "inherit" by walking back up the nodes
    # fast path optimization - if it's a valid float, don't
    # try to parse it.
    try:
        return float(val)
    except ValueError:
        return valueToPixels(val)


def fractionalValue(value):
    """ Parse a string consisting of a float in the range [0..1] or a
    percentage as a float number in the range [0..1].
    """
    if value.endswith("%"):
        return float(value[:-1]) / 100.0
    else:
        return float(value)


units_to_px = {
    "in": 72,
    "pc": 12,
    "cm": 72.0 / 2.54,
    "mm": 72.0 / 25.4,
    "px": 1,
    "pt": 1,
}


def valueToPixels(val, defaultUnits="px"):
    # This pretends that 1px == 1pt. For our purposes, that's fine since we
    # don't actually care about px. The name of the function is bad.
    # TODO: refactor in order to handle relative percentages and em and ex.
    # TODO: manage default units
    from pyparsing import ParseException

    if val.endswith("%"):
        # TODO: this is one of those relative values we need to fix.
        return float(val[:-1]) / 100.0
    try:
        val, unit = values.length.parseString(val)
    except ParseException:
        raise
    val *= units_to_px.get(unit, 1)
    return val


def pathHandler(func):
    """decorator for methods which return a path operation
        Creates the path they will fill,
        and generates the path operations for the node
    """

    @wraps(func)
    def inner(self, node):
        path = self.renderer.makePath()
        results = func(self, node, path)

        ops = [(self.renderer.pushState, ())]

        # the results willbe None, unless the path has something which affects
        # the render stack, such as clipping
        if results is not None:
            cpath, cops = results
            path = cpath
            ops = cops + ops

        ops.extend(self.createTransformOpsFromNode(node))
        ops.extend(self.generatePathOps(path))
        ops.append((self.renderer.popState, ()))
        return path, ops

    return inner


class ResourceGetter(object):
    """ Simple context for getting relative-pathed resources.
    """

    def __init__(self, dirname=None):
        if dirname is None:
            dirname = os.getcwd()
        self.dirname = dirname

    @classmethod
    def fromfilename(cls, filename):
        """ Use the directory containing this file as the base directory.
        """
        dirname = os.path.abspath(os.path.dirname(filename))
        return cls(dirname)

    def newbase(self, dirname):
        """ Return a new ResourceGetter using a new base directory found
        relative to this one.
        """
        dirname = os.path.abspath(os.path.join(self.dirname, dirname))
        return self.__class__(dirname)

    def resolve(self, path):
        """ Resolve a path and the associated opener function against this
        context.
        """
        scheme, netloc, path_part, _, _, _ = urlparse.urlparse(path)
        if scheme not in ("", "file"):
            # Plain URI. Pass it back.
            # Read the data and stuff it in a StringIO in order to satisfy
            # functions that need a functioning seek() and stuff.
            return (
                path,
                lambda uri: BytesIO(urllib.request.urlopen(uri).read()),
            )
        path = os.path.abspath(os.path.join(self.dirname, path_part))
        return path, lambda fn: open(fn, "rb")

    def open_svg(self, path):
        """ Resolve and read an SVG file into an Element.
        """
        path, open = self.resolve(path)
        f = open(path)
        tree = ET.parse(f)
        element = tree.getroot()
        return element

    def open_image(self, path):
        """ Resolve and read an image into an appropriate object for the
        renderer.

        """
        import numpy
        from PIL import Image

        path, open = self.resolve(path)
        with open(path) as fin:
            pil_img = Image.open(fin)

        if pil_img.mode not in ("RGB", "RGBA"):
            pil_img = pil_img.convert("RGBA")
        img = numpy.asarray(pil_img)
        return img


class SVGDocument(object):
    def __init__(self, element, resources=None, renderer=NullRenderer):
        """
        Create an SVG document from an ElementTree node.

        FIXME: this is really wrong that the doc must know about the renderer
        """
        self.renderer = renderer

        self.lastControl = None
        self.brushCache = {}
        self.penCache = {}

        self.handlers = {
            SVG.svg: self.addGroupToDocument,
            SVG.a: self.addGroupToDocument,
            SVG.g: self.addGroupToDocument,
            SVG.symbol: self.addGroupToDocument,
            SVG.use: self.addUseToDocument,
            SVG.switch: self.addSwitchToDocument,
            SVG.image: self.addImageToDocument,
            SVG.rect: self.addRectToDocument,
            SVG.circle: self.addCircleToDocument,
            SVG.ellipse: self.addEllipseToDocument,
            SVG.line: self.addLineToDocument,
            SVG.polyline: self.addPolyLineToDocument,
            SVG.polygon: self.addPolygonToDocument,
            SVG.path: self.addPathDataToDocument,
            SVG.text: self.addTextToDocument,
        }

        assert element.tag == SVG.svg, "Not an SVG fragment"
        if resources is None:
            resources = ResourceGetter()
        self.resources = resources

        self.tree = element
        # Mapping of (URI, XML id) pairs to elements. '' is the URI for local
        # resources. Use self.update(findIDs(element), uri) for adding elements
        # from other URIs.
        self.idmap = self.findIDs(element)
        self.paths = {}
        self.stateStack = [{}]
        self.clippingStack = []
        path, ops = self.processElement(element)
        self.ops = ops

    @classmethod
    def createFromFile(cls, filename, renderer):
        if not os.path.exists(filename):
            raise IOError("No such file: " + filename)

        tree = ET.parse(filename)
        root = tree.getroot()

        resources = ResourceGetter(os.path.dirname(filename))
        return cls(root, resources, renderer)

    def getSize(self):
        width = -1
        width_node = self.tree.get("width")
        if width_node is not None:
            if width_node.endswith("cm"):
                # assumes dpi of 72
                width = int(float(width_node.split("cm")[0]) * 72 / 2.54)
            else:
                # omit 'px' if it was specified
                width = int(float(width_node.split("px")[0]))

        height = -1
        height_node = self.tree.get("height")
        if height_node is not None:
            if height_node.endswith("cm"):
                # assumes dpi of 72
                height = int(float(height_node.split("cm")[0]) * 72 / 2.54)
            else:
                # omit 'px' if it was specified
                height = int(float(height_node.split("px")[0]))

        return (width, height)

    def findIDs(self, element, uri=""):
        """ Iterate through the tree under an element and record all elements
        which specify an id attribute.

        The root element is given the ID '' in addition to whatever id=
        attribute it may be given.
        """
        idmap = {}
        for e in element.iter():
            id = e.get("id", None)
            if id is not None:
                idmap[(uri, id)] = e
        idmap[(uri, "")] = element
        return idmap

    def dereference(self, href, resources=None):
        """ Find the element specified by the give href.

        Parameters
        ----------
        href : str
            The reference pointing to the desired element. Forms like
            'url(uri#theid)' and 'uri#xpointer(id(theid))' will be normalized
            to pull out just the uri and the id.
        resources : ResourceGetter, optional
            The ResourceGetter to use. If not provided, the one attached to the
            SVGDocument is used. This is useful when silly test suites like to
            test annoying XLink features.
            FIXME: <sigh> xml:base is inheritable.

        Returns
        -------
        element : Element

        Raises
        ------
        KeyError :
            If the element is not found.
        """
        uri, fragment = normalize_href(href)
        if uri and (uri, fragment) not in self.idmap:
            # Record all of the IDed elements in the referenced document.
            if resources is None:
                resources = self.resources
            element = resources.open_svg(uri)
            self.idmap.update(self.findIDs(element, uri))
        return self.idmap[(uri, fragment)]

    @property
    def state(self):
        """ Retrieve the current state, without popping"""
        return self.stateStack[-1]

    def getLocalState(self, element, state=None):
        """ Get the state local to an element.
        """
        if state is None:
            state = self.state
        current = dict(state)
        element_items = [
            (k, v) for (k, v) in element.items() if v != "inherit"
        ]
        current.update(element_items)
        style_items = [
            (k, v)
            for (k, v) in css.inlineStyle(element.get("style", "")).items()
            if v != "inherit"
        ]
        current.update(style_items)
        return current

    def processElement(self, element):
        """ Process one element of the XML tree.
        Returns the path representing the node,
        and an operation list for drawing the node.

        Parent nodes should return a path (for hittesting), but
        no draw operations
        """
        current = self.getLocalState(element)
        self.stateStack.append(current)
        handler = self.handlers.get(element.tag, lambda *any: (None, None))
        path, ops = handler(element)
        self.paths[element] = path
        self.stateStack.pop()
        return path, ops

    def createTransformOpsFromNode(self, node, attribute="transform"):
        """ Returns an oplist for transformations.
        This applies to a node, not the current state because
        the transform stack is saved in the graphics context.

        This oplist does *not* include the push/pop state commands
        """
        ops = []
        transform = node.get(attribute, None)
        # todo: replace this with a mapping list
        if transform:
            for transform, args in css.transformList.parseString(transform):
                if transform == "scale":
                    if len(args) == 1:
                        x = y = args[0]
                    else:
                        x, y = args
                    ops.append((self.renderer.scale, (x, y)))
                if transform == "translate":
                    if len(args) == 1:
                        x = args[0]
                        y = 0
                    else:
                        x, y = args
                    ops.append((self.renderer.translate, (x, y)))
                if transform == "rotate":
                    if len(args) == 3:
                        angle, cx, cy = args
                        angle = math.radians(angle)
                        ops.extend(
                            [
                                (self.renderer.translate, (cx, cy)),
                                (self.renderer.rotate, (angle,)),
                                (self.renderer.translate, (-cx, -cy)),
                            ]
                        )
                    else:
                        angle = args[0]
                        angle = math.radians(angle)
                        ops.append((self.renderer.rotate, (angle,)))
                if transform == "matrix":
                    matrix = self.renderer.createAffineMatrix(*args)
                    ops.append((self.renderer.concatTransform, (matrix,)))
                if transform == "skewX":
                    matrix = self.renderer.createAffineMatrix(
                        1, 0, math.tan(math.radians(args[0])), 1, 0, 0
                    )
                    ops.append((self.renderer.concatTransform, (matrix,)))
                if transform == "skewY":
                    matrix = self.renderer.createAffineMatrix(
                        1, math.tan(math.radians(args[0])), 0, 1, 0, 0
                    )
                    ops.append((self.renderer.concatTransform, (matrix,)))
        return ops

    def createTransformOpsFromXY(self, node):
        """ On some nodes, x and y attributes cause a translation of the
        coordinate system.
        """
        ops = []
        # Now process x,y attributes. Per 7.6 of the SVG1.1 spec, these are
        # interpreted after transform=.
        x = attrAsFloat(node, "x")
        y = attrAsFloat(node, "y")
        if x != 0.0 or y != 0.0:
            ops.append((self.renderer.translate, (x, y)))
        return ops

    def addGroupToDocument(self, node):
        """ For parent elements: push on a state,
        then process all child elements
        """
        ops = [(self.renderer.pushState, ())]

        path = self.renderer.makePath()
        ops.extend(self.createTransformOpsFromNode(node))
        ops.extend(self.createTransformOpsFromXY(node))
        for child in node:
            cpath, cops = self.processElement(child)
            if cpath:
                path.AddPath(cpath)
            if cops:
                ops.extend(cops)
        ops.append((self.renderer.popState, ()))
        return path, ops

    def addUseToDocument(self, node):
        """ Add a <use> tag to the document.
        """
        # FIXME: width,height?
        # FIXME: this could lead to circular references in erroneous documents.
        # It would be nice to raise an exception in this case.
        href = node.get(XLink.href, None)
        if href is None:
            # Links to nothing.
            return None, []
        base = self.state.get(XML.base, None)
        if base is not None:
            resources = self.resources.newbase(base)
        else:
            resources = self.resources
        try:
            element = self.dereference(href, resources)
        except (OSError, IOError) as e:
            # SVG file cannot be found.
            warnings.warn(
                "Could not find SVG file %s. %s: %s"
                % (href, e.__class__.__name__, e)
            )
            return None, []

        ops = [(self.renderer.pushState, ())]

        path = self.renderer.makePath()
        ops.extend(self.createTransformOpsFromNode(node))
        ops.extend(self.createTransformOpsFromXY(node))
        cpath, cops = self.processElement(element)
        if cpath:
            path.AddPath(cpath)
        if cops:
            ops.extend(cops)
        ops.append((self.renderer.popState, ()))
        return path, ops

    def addSwitchToDocument(self, node):
        """ Process a <switch> tag.
        """
        for child in node:
            if child.get("requiredExtensions") is None:
                # This renderer does not support any extensions. Pick the first
                # item that works. This allows us to read SVG files made with
                # Adobe Illustrator. They embed the SVG content in a <switch>
                # along with an encoded representation in AI format. The
                # encoded non-SVG bit has a requiredExtensions= attribute.
                # FIXME: other tests?
                return self.processElement(child)
        return None, None

    def addImageToDocument(self, node):
        """ Add an <image> tag to the document.
        """
        href = node.get(XLink.href, None)
        if href is None:
            # Links to nothing.
            return None, []
        base = self.state.get(XML.base, None)
        if base is not None:
            resources = self.resources.newbase(base)
        else:
            resources = self.resources
        uri, fragment = normalize_href(href)
        if uri.endswith(".svg") and not uri.startswith("data:"):
            # FIXME: Pretend it's a <use>.
            return self.addUseToDocument(node)
        try:
            image = resources.open_image(uri)
        except (OSError, IOError) as e:
            # Image cannot be found.
            warnings.warn(
                "Could not find image file %s. %s: %s"
                % (uri[:100], e.__class__.__name__, str(e)[:100])
            )
            return None, []
        ops = [(self.renderer.pushState, ())]
        ops.extend(self.createTransformOpsFromNode(node))
        if type(image).__name__ == "Element":
            # FIXME: bad API. Bad typecheck since ET.Element is a factory
            # function, not a type.
            # This is an SVG file, not an image.
            imgpath, imgops = self.processElement(image)
            ops.extend(imgops)
            ops.append((self.renderer.popState, ()))
            return imgpath, ops
        x = attrAsFloat(node, "x")
        y = attrAsFloat(node, "y")
        width = attrAsFloat(node, "width")
        height = attrAsFloat(node, "height")
        if width == 0.0 or height == 0.0:
            return None, []
        ops.extend(
            [
                (self.renderer.DrawImage, (image, x, y, width, height)),
                (self.renderer.popState, ()),
            ]
        )
        return None, ops

    def getFontFromState(self):
        font = self.renderer.getFont()
        family = self.state.get("font-family")
        if family:
            font.face_name = family

        style = self.state.get("font-style")
        if style:
            self.renderer.setFontStyle(font, style)

        weight = self.state.get("font-weight")
        if weight:
            self.renderer.setFontWeight(font, weight)

        size = self.state.get("font-size")
        # TODO: properly handle inheritance.
        if size and size != "inherit":
            val, unit = values.length.parseString(size)
            self.renderer.setFontSize(font, val)

        # fixme: Handle text-decoration for line-through and underline.
        #        These are probably done externally using drawing commands.
        return font

    def addTextToDocument(self, node):
        # TODO: these attributes can actually be lists of numbers.
        # text-text-04-t.svg
        x, y = [attrAsFloat(node, attr) for attr in ("x", "y")]

        font = self.getFontFromState()
        brush = self.getBrushFromState()

        if not (brush and hasattr(brush, "IsOk") and brush.IsOk()):
            black_tuple = (255, 255, 255, 255)
            brush = self.renderer.createBrush(black_tuple)
        # TODO: handle <tspan>, <a> and <tref>.
        # TODO: handle xml:space="preserve"? The following more or less
        # corresponds to xml:space="default".
        if node.text:
            text = " ".join(node.text.split())
        else:
            text = ""
        if text is None:
            return None, []
        text_anchor = self.state.get("text-anchor", "start")
        ops = [(self.renderer.pushState, ())]
        ops.extend(self.createTransformOpsFromNode(node))
        ops.extend(
            [
                (self.renderer.setFont, (font, brush)),
                (self.renderer.DrawText, (text, x, y, brush, text_anchor)),
                (self.renderer.popState, ()),
            ]
        )
        return None, ops

    @pathHandler
    def addRectToDocument(self, node, path):
        x, y, w, h = (
            attrAsFloat(node, attr) for attr in ["x", "y", "width", "height"]
        )
        rx = node.get("rx")
        ry = node.get("ry")

        ops = []

        if "clip-path" in node:
            element = self.dereference(node.get("clip-path"))

            ops = [(self.renderer.pushState, ())]

            clip_path = self.renderer.makePath()
            ops.extend(self.createTransformOpsFromNode(element))
            ops.extend(self.generatePathOps(clip_path))
            for child in element:
                cpath, cops = self.processElement(child)
                if cpath:
                    clip_path.AddPath(cpath)
                    ops.append((self.renderer.clipPath, (clip_path,)))
                    path.AddPath(clip_path)
                if cops:
                    ops.extend(cops)
            ops.append((self.renderer.popState, ()))

        if not (w and h):
            path.MoveToPoint(x, y)  # keep the current point correct
            return
        if rx or ry:
            if rx and ry:
                rx, ry = float(rx), float(ry)
            elif rx:
                rx = ry = float(rx)
            elif ry:
                rx = ry = float(ry)
            # value clamping as per spec section 9.2
            rx = min(rx, w / 2)
            ry = min(ry, h / 2)

            path.AddRoundedRectangleEx(x, y, w, h, rx, ry)
        else:
            if len(self.clippingStack) > 0:
                self.renderer.clipPath()
            else:
                path.AddRectangle(x, y, w, h)

        return path, ops

    @pathHandler
    def addCircleToDocument(self, node, path):
        cx, cy, r = [attrAsFloat(node, attr) for attr in ("cx", "cy", "r")]
        path.AddCircle(cx, cy, r)

    @pathHandler
    def addEllipseToDocument(self, node, path):
        cx, cy, rx, ry = [
            float(node.get(attr, 0)) for attr in ("cx", "cy", "rx", "ry")
        ]
        # cx, cy are centerpoint.
        # rx, ry are radius.
        if rx <= 0 or ry <= 0:
            return
        path.AddEllipse(cx, cy, rx, ry)

    @pathHandler
    def addLineToDocument(self, node, path):
        x1, y1, x2, y2 = [
            attrAsFloat(node, attr) for attr in ("x1", "y1", "x2", "y2")
        ]
        path.MoveToPoint(x1, y1)
        path.AddLineToPoint(x2, y2)

    @pathHandler
    def addPolyLineToDocument(self, node, path):
        # translate to pathdata and render that
        data = "M " + node.get("points")
        self.addPathDataToPath(data, path)

    @pathHandler
    def addPolygonToDocument(self, node, path):
        # translate to pathdata and render that
        points = node.get("points")
        if points is not None:
            data = "M " + points + " Z"
            self.addPathDataToPath(data, path)

    @pathHandler
    def addPathDataToDocument(self, node, path):
        self.addPathDataToPath(node.get("d", ""), path)

    def addPathDataToPath(self, data, path):
        self.lastControl = None
        self.lastControlQ = None
        self.firstPoints = []

        def normalizeStrokes(parseResults):
            """ The data comes from the parser in the
            form of (command, [list of arguments]).
            We translate that to [(command, args[0]), (command, args[1])]
            via a generator.

            M is special cased because its subsequent arguments
            become linetos.
            """
            for command, arguments in parseResults:
                if not arguments:
                    yield (command, ())
                else:
                    arguments = iter(arguments)
                    if command == "m":
                        yield (command, next(arguments))
                        command = "l"
                    elif command == "M":
                        yield (command, next(arguments))
                        command = "L"
                    for arg in arguments:
                        yield (command, arg)

        try:
            parsed = svg_parser.parse(data)
        except SyntaxError as e:
            print("SyntaxError: %s" % e)
            print("data = %r" % data)
        else:
            for stroke in normalizeStrokes(parsed):
                self.addStrokeToPath(path, stroke)

    def generatePathOps(self, path):
        """ Look at the current state and generate the
        draw operations (fill, stroke, neither) for the path.
        """
        ops = []
        brush = self.getBrushFromState(path)
        fillRule = self.state.get("fill-rule", "nonzero")
        fr = self.renderer.fill_rules.get(fillRule)
        if brush is not None:
            if isinstance(brush, AbstractGradientBrush):
                ops.extend([(self.renderer.gradientPath, (path, brush))])
            else:
                ops.extend(
                    [
                        (self.renderer.setBrush, (brush,)),
                        (self.renderer.fillPath, (path, fr)),
                    ]
                )
        pen = self.getPenFromState()
        if pen is not None:
            ops.extend(
                [
                    (self.renderer.setPen, (pen,)),
                    (self.renderer.strokePath, (path,)),
                ]
            )
        return ops

    def getPenFromState(self):
        pencolour = self.state.get("stroke", "none")
        if pencolour == "currentColor":
            pencolour = self.state.get("color", "none")
        if pencolour == "transparent":
            return self.renderer.TransparentPen
        if pencolour == "none":
            return self.renderer.NullPen
        type, value = colourValue.parseString(pencolour)
        if type == "URL":
            warnings.warn("Color servers for stroking not implemented")
            return self.renderer.NullPen
        else:
            if value[:3] == (-1, -1, -1):
                return self.renderer.NullPen
            pen = self.renderer.createPen(value)
        width = self.state.get("stroke-width")
        if width:
            width, units = values.length.parseString(width)
            pen.SetWidth(width)
        stroke_dasharray = self.state.get("stroke-dasharray", "none")
        if stroke_dasharray != "none":
            stroke_dasharray = list(
                map(valueToPixels, stroke_dasharray.replace(",", " ").split())
            )
            if len(stroke_dasharray) % 2:
                # Repeat to get an even array.
                stroke_dasharray = stroke_dasharray * 2
            stroke_dashoffset = valueToPixels(
                self.state.get("stroke-dashoffset", "0")
            )
            self.renderer.setPenDash(pen, stroke_dasharray, stroke_dashoffset)
        pen.SetCap(
            self.renderer.caps.get(
                self.state.get("stroke-linecap", None),
                self.renderer.caps["butt"],
            )
        )
        pen.SetJoin(
            self.renderer.joins.get(
                self.state.get("stroke-linejoin", None),
                self.renderer.joins["miter"],
            )
        )
        return self.renderer.createNativePen(pen)

    def parseStops(self, element):
        """ Parse the color stops from a gradient definition.
        """
        stops = []
        gradient_state = self.getLocalState(element)
        for stop in element:
            if stop.tag != SVG.stop:
                warnings.warn(
                    "Skipping non-<stop> element <%s> in <%s>"
                    % (stop.tag, element.tag)
                )
                continue
            stopstate = self.getLocalState(stop, gradient_state)
            offset = fractionalValue(stop.get("offset"))
            offset = max(min(offset, 1.0), 0.0)
            default_opacity = "1"
            color = stopstate.get("stop-color", "black")
            if color in ["inherit", "currentColor"]:
                # Try looking up in the gradient element itself.
                # FIXME: Look farther up?
                color = stopstate.get("color", "black")
            elif color == "none":
                color = "black"
                default_opacity = "0"
            type, color = colourValue.parseString(color)
            if type == "URL":
                warnings.warn("Color servers for gradients not implemented")
            elif color[:3] == (-1, -1, -1):
                # FIXME: is this right?
                color = (0.0, 0.0, 0.0, 0.0)
            opacity = stopstate.get("stop-opacity", default_opacity)
            if opacity == "inherit":
                # FIXME: what value to inherit?
                opacity = "1"
            opacity = float(opacity)
            row = (
                offset,
                color[0] / 255.0,
                color[1] / 255.0,
                color[2] / 255.0,
                opacity,
            )
            stops.append(row)
        stops.sort()
        if len(stops) == 0:
            return numpy.array([])
        if stops[0][0] > 0.0:
            stops.insert(0, (0.0,) + stops[0][1:])
        if stops[-1][0] < 1.0:
            stops.append((1.0,) + stops[-1][1:])
        return numpy.transpose(stops)

    def getBrushFromState(self, path=None):
        brushcolour = self.state.get("fill", "black").strip()
        type, details = paintValue.parseString(brushcolour)
        if type == "URL":
            url, fallback = details
            url = urlparse.urlunsplit(url)
            try:
                element = self.dereference(url)
            except ParseError:
                element = None
            if element is None:
                if fallback:
                    type, details = fallback
                else:
                    r, g, b, = 0, 0, 0
            else:
                # The referencing tag controls the kind of gradient. Mostly,
                # it's just the stops that are pulled from the referenced
                # gradient tag.
                element_tag = element.tag
                if element_tag not in (SVG.linearGradient, SVG.radialGradient):
                    if "}" in element_tag:
                        element_tag[element_tag.find("}") + 1:]
                    warnings.warn("<%s> not implemented" % element_tag)
                    return self.renderer.NullBrush
                href = element.get(XLink.href, None)
                seen = set([element])
                # The attributes on the referencing element override those on
                # the referenced element.
                state = dict(element.items())
                while href is not None:
                    # Gradient is a reference.
                    element = self.dereference(href)
                    if element in seen:
                        # FIXME: if they are loaded from another file, will
                        # element identity work correctly?
                        raise ValueError(
                            "Element referred to by %r is a "
                            "circular reference." % href
                        )
                    seen.add(element)
                    # new_state = dict(element.items())
                    # new_state.update(state)
                    # state = new_state
                    href = element.get(XLink.href, None)
                spreadMethod = state.get("spreadMethod", "pad")
                transforms = self.createTransformOpsFromNode(
                    state, "gradientTransform"
                )
                if not transforms:
                    transforms = []
                units = state.get("gradientUnits", "objectBoundingBox")
                stops = self.parseStops(element)
                if stops.size == 0:
                    return self.renderer.NullBrush
                if element_tag == SVG.linearGradient:
                    x1 = attrAsFloat(state, "x1", "0%")
                    y1 = attrAsFloat(state, "y1", "0%")
                    x2 = attrAsFloat(state, "x2", "100%")
                    y2 = attrAsFloat(state, "y2", "0%")
                    return self.renderer.createLinearGradientBrush(
                        x1,
                        y1,
                        x2,
                        y2,
                        stops,
                        spreadMethod=spreadMethod,
                        transforms=transforms,
                        units=units,
                    )
                elif element_tag == SVG.radialGradient:
                    cx = attrAsFloat(state, "cx", "50%")
                    cy = attrAsFloat(state, "cy", "50%")
                    r = attrAsFloat(state, "r", "50%")
                    fx = attrAsFloat(state, "fx", state.get("cx", "50%"))
                    fy = attrAsFloat(state, "fy", state.get("cy", "50%"))
                    return self.renderer.createRadialGradientBrush(
                        cx,
                        cy,
                        r,
                        stops,
                        fx,
                        fy,
                        spreadMethod=spreadMethod,
                        transforms=transforms,
                        units=units,
                    )
                else:
                    # invlid gradient specified
                    return self.renderer.NullBrush
            r, g, b = 0, 0, 0
        if type == "CURRENTCOLOR":
            type, details = paintValue.parseString(
                self.state.get("color", "none")
            )
        if type == "RGB":
            r, g, b = details
        elif type == "NONE":
            return self.renderer.NullBrush
        opacity = self.state.get(
            "fill-opacity", self.state.get("opacity", "1")
        )
        opacity = float(opacity)
        opacity = min(max(opacity, 0.0), 1.0)
        a = 255 * opacity
        # using try/except block instead of
        # just setdefault because the brush and colour would
        # be created every time anyway in order to pass them,
        # defeating the purpose of the cache
        try:
            brush = self.brushCache[(r, g, b, a)]
        except KeyError:
            brush = self.brushCache.setdefault(
                (r, g, b, a), self.renderer.createBrush((r, g, b, a))
            )
        return brush

    def addStrokeToPath(self, path, stroke):
        """ Given a stroke from a path command
        (in the form (command, arguments)) create the path
        commands that represent it.

        TODO: break out into (yet another) class/module,
        especially so we can get O(1) dispatch on type?
        """
        type, arg = stroke
        relative = False
        if type == type.lower():
            relative = True
            ox, oy = path.GetCurrentPoint()
        else:
            ox = oy = 0

        def normalizePoint(arg):
            x, y = arg
            return x + ox, y + oy

        def reflectPoint(point, relativeTo):
            x, y = point
            a, b = relativeTo
            return ((a * 2) - x), ((b * 2) - y)

        type = type.upper()
        if type == "M":
            pt = normalizePoint(arg)
            self.firstPoints.append(pt)
            path.MoveToPoint(*pt)
        elif type == "L":
            pt = normalizePoint(arg)
            path.AddLineToPoint(*pt)
        elif type == "C":
            control1, control2, endpoint = map(normalizePoint, arg)
            self.lastControl = control2
            path.AddCurveToPoint(control1, control2, endpoint)

        elif type == "S":
            control2, endpoint = map(normalizePoint, arg)
            if self.lastControl:
                control1 = reflectPoint(
                    self.lastControl, path.GetCurrentPoint()
                )
            else:
                control1 = path.GetCurrentPoint()
            self.lastControl = control2
            path.AddCurveToPoint(control1, control2, endpoint)
        elif type == "Q":
            (cx, cy), (x, y) = map(normalizePoint, arg)
            self.lastControlQ = (cx, cy)
            path.AddQuadCurveToPoint(cx, cy, x, y)
        elif type == "T":
            x, y, = normalizePoint(arg)
            if self.lastControlQ:
                cx, cy = reflectPoint(
                    self.lastControlQ, path.GetCurrentPoint()
                )
            else:
                cx, cy = path.GetCurrentPoint()
            self.lastControlQ = (cx, cy)
            path.AddQuadCurveToPoint(cx, cy, x, y)

        elif type == "V":
            _, y = normalizePoint((0, arg))
            x, _ = path.GetCurrentPoint()
            path.AddLineToPoint(x, y)

        elif type == "H":
            x, _ = normalizePoint((arg, 0))
            _, y = path.GetCurrentPoint()
            path.AddLineToPoint(x, y)

        elif type == "A":
            (
                (rx, ry),  # radii of ellipse
                angle,  # angle of rotation on the ellipse in degrees
                large_arc_flag,
                sweep_flag,  # arc and stroke angle flags
                (x2, y2),  # endpoint on the arc
            ) = arg

            x2, y2 = normalizePoint((x2, y2))

            path.elliptical_arc_to(
                rx, ry, angle, large_arc_flag, sweep_flag, x2, y2
            )

        elif type == "Z":
            # ~ Bugginess:
            # ~ CloseSubpath() doesn't change the
            # ~ current point, as SVG spec requires.
            # ~ However, manually moving to the endpoint afterward opens a new
            # ~ subpath and (apparently) messes with stroked but not filled
            # ~ paths.
            # ~ This is possibly a bug in GDI+?
            # ~ Manually closing the path via AddLineTo gives incorrect line
            # ~ join results
            # ~ Manually closing the path *and* calling CloseSubpath() appears
            # ~ to give correct results on win32

            path.CloseSubpath()

    def render(self, context):
        if not hasattr(self, "ops"):
            return
        for op, args in self.ops:
            op(context, *args)

