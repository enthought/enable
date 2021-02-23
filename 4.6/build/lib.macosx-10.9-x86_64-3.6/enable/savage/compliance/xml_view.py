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
""" Traits UI tools for viewing the XML tree of SVG files.
"""


from traits.api import HasTraits, List, Property, Str
from traitsui.api import Item, ModelView, TreeEditor, TreeNode, View


known_namespaces = {
    "{http://www.w3.org/2000/svg}": "svg",
    "{http://www.w3.org/2000/02/svg/testsuite/description/}": "testcase",
    "{http://www.w3.org/1999/xlink}": "xlink",
    "{http://www.w3.org/XML/1998/namespace}": "xml",
    "{http://sodipodi.sourceforge.net/DTD/sodipodi-0.dtd}": "sodipodi",
    "{http://inkscape.sourceforge.net/DTD/sodipodi-0.dtd}": "sodipodi",
    "{http://purl.org/dc/elements/1.1/}": "dc",
    "{http://web.resource.org/cc/}": "cc",
    "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}": "rdf",
    "{http://www.inkscape.org/namespaces/inkscape}": "inkscape",
    "{http://ns.adobe.com/AdobeIllustrator/10.0/}": "adobei",
    "{http://ns.adobe.com/AdobeSVGViewerExtensions/3.0/}": "adobea",
    "{http://ns.adobe.com/Graphs/1.0/}": "graphs",
    "{http://ns.adobe.com/Extensibility/1.0/}": "adobex",
}


def normalize_name(name):
    """ Normalize XML names to abbreviate namespaces.
    """
    for ns in known_namespaces:
        if name.startswith(ns):
            name = "%s:%s" % (known_namespaces[ns], name[len(ns):])
    return name


class Attribute(HasTraits):
    """ View model for an XML attribute.
    """

    name = Str()
    value = Str()

    label = Property()

    def _get_label(self):
        return "%s : %s" % (normalize_name(self.name), self.value)


class Element(HasTraits):
    """ View model for an XML element.
    """

    tag = Str()
    children = List()
    attributes = List(Attribute)
    text = Str()

    label = Property()
    kids = Property()

    def _get_label(self):
        return normalize_name(self.tag)

    def _get_kids(self):
        kids = self.children + self.attributes
        if self.text:
            kids.append(Attribute(value=self.text))
        return kids


def xml_to_tree(root):
    """ Convert an ElementTree Element to the view models for viewing in the
    TreeEditor.
    """
    element = Element(tag=root.tag)
    if root.text is not None:
        element.text = root.text.strip()
    for name in sorted(root.keys()):
        element.attributes.append(Attribute(name=name, value=root.get(name)))
    for child in root:
        element.children.append(xml_to_tree(child))
    return element


xml_tree_editor = TreeEditor(
    nodes=[
        TreeNode(
            node_for=[Element], children="kids", label="label", menu=False
        ),
        TreeNode(
            node_for=[Attribute], children="", label="label", menu=False
        ),
    ],
    editable=False,
    show_icons=False,
)


class XMLTree(ModelView):
    """ Handler for viewing XML trees.
    """

    traits_view = View(
        Item("model", editor=xml_tree_editor, show_label=False),
        width=1024,
        height=768,
        resizable=True,
    )

    @classmethod
    def fromxml(cls, root, **traits):
        return cls(model=xml_to_tree(root), **traits)


def main():
    from xml.etree import cElementTree as ET
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("file")

    args = parser.parse_args()
    xml = ET.parse(args.file).getroot()
    t = XMLTree.fromxml(xml)
    t.configure_traits()


if __name__ == "__main__":
    main()
