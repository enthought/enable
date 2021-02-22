# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
from time import perf_counter

from kiva.agg import AffineMatrix, GraphicsContextArray
from kiva.api import MODERN, Font

gc = GraphicsContextArray((200, 200))

font = Font(family=MODERN)
# print(font.size)
font.size = 8
gc.set_font(font)


t1 = perf_counter()

# consecutive printing of text.
with gc:
    gc.set_antialias(False)
    gc.set_fill_color((0, 1, 0))
    gc.translate_ctm(50, 50)
    gc.rotate_ctm(3.1416 / 4)
    gc.show_text("hello")
    gc.translate_ctm(-50, -50)
    gc.set_text_matrix(AffineMatrix())
    gc.set_fill_color((0, 1, 1))
    gc.show_text("hello")

t2 = perf_counter()
print("aliased:", t2 - t1)
gc.save("text_aliased.bmp")

gc = GraphicsContextArray((200, 200))

font = Font(family=MODERN)
# print(font.size)
font.size = 8
gc.set_font(font)

t1 = perf_counter()

with gc:
    gc.set_antialias(True)
    gc.set_fill_color((0, 1, 0))
    gc.translate_ctm(50, 50)
    gc.rotate_ctm(3.1416 / 4)
    gc.show_text("hello")
    gc.translate_ctm(-50, -50)
    gc.set_text_matrix(AffineMatrix())
    gc.set_fill_color((0, 1, 1))
    gc.show_text("hello")

t2 = perf_counter()
print("antialiased:", t2 - t1)
gc.save("text_antialiased.bmp")

"""
with gc:
    gc.set_fill_color((0,1,0))
    gc.rotate_ctm(-45)
    gc.show_text_at_point("hello")
    gc.set_fill_color((0,1,1))
    gc.show_text("hello")

with gc:
    gc.translate_ctm(80,-3)
    gc.show_text("hello")

with gc:
    gc.set_fill_color((0,1,0))
    gc.translate_ctm(50,50)
    gc.show_text("hello")

with gc:
    gc.set_fill_color((0,1,0))
    gc.translate_ctm(50,50)
    gc.rotate_ctm(3.1416/4)
    gc.show_text("hello")
    gc.set_fill_color((1,0,0))
    gc.show_text("hello")
"""


gc.save("text_antiliased.bmp")
