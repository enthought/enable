import unittest
import enthought.savage.svg.document as document
import xml.etree.cElementTree as etree
from cStringIO import StringIO

from enthought.savage.svg.backends.kiva.renderer import Renderer as KivaRenderer

minimalSVG = etree.parse(StringIO(r"""<?xml version="1.0" standalone="no"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN"
  "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
<svg xmlns="http://www.w3.org/2000/svg" version="1.1"></svg>"""))

class TestBrushFromColourValue(unittest.TestCase):

    def setUp(self):
        self.document = document.SVGDocument(minimalSVG.getroot(), renderer=KivaRenderer())
        self.stateStack = [{}]

    def testNone(self):
        self.document.state["fill"] = 'none'
        self.assertEqual(
            self.document.getBrushFromState(),
            None
        )

    def testCurrentColour(self):
        self.document.state["fill"] = 'currentColor'
        self.document.state["color"] = "rgb(100,100,100)"
        self.assertEqual(
            self.document.getBrushFromState().color,
            (100,100,100,255)
        )

    def testCurrentColourNull(self):
        self.document.state["fill"] = 'currentColor'
        self.assertEqual(
            self.document.getBrushFromState(),
            None
        )

    def testOpacity(self):
        self.document.state["fill"] = 'rgb(255,100,10)'
        self.document.state["fill-opacity"] = 0.5
        self.assertEqual(
            self.document.getBrushFromState().color[-1],
            127.5
        )

    def testOpacityClampHigh(self):
        self.document.state["fill"] = 'rgb(255,100,10)'
        self.document.state["fill-opacity"] = 5
        self.assertEqual(
            self.document.getBrushFromState().color[-1],
            255
        )

    def testOpacityClampLow(self):
        self.document.state["fill"] = 'rgb(255,100,10)'
        self.document.state["fill-opacity"] = -100
        self.assertEqual(
            self.document.getBrushFromState().color[-1],
            0
        )
    def testURLFallback(self):
        self.document.state["fill"] = "url(http://google.com) red"
        self.assertEqual(
            self.document.getBrushFromState().color,
            (255,0,0)
        )

class TestValueToPixels(unittest.TestCase):
    """ Make sure that CSS length values get converted correctly to pixels"""
    def testDefault(self):
        got = document.valueToPixels("12")
        self.assertEqual(got, 12)
    def testPointConversion(self):
        got = document.valueToPixels('14pt')
        self.assertEqual(got, 22)
