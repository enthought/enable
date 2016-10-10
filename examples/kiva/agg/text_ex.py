

import time
from kiva.fonttools import Font
from kiva.constants import MODERN
from kiva.agg import AffineMatrix, GraphicsContextArray

gc = GraphicsContextArray((200,200))

font = Font(family=MODERN)
#print font.size
font.size=8
gc.set_font(font)


t1 = time.clock()

# consecutive printing of text.
with gc:
    gc.set_antialias(False)
    gc.set_fill_color((0,1,0))
    gc.translate_ctm(50,50)
    gc.rotate_ctm(3.1416/4)
    gc.show_text("hello")
    gc.translate_ctm(-50,-50)
    gc.set_text_matrix(AffineMatrix())
    gc.set_fill_color((0,1,1))
    gc.show_text("hello")

t2 = time.clock()
print('aliased:', t2 - t1)
gc.save("text_aliased.bmp")

gc = GraphicsContextArray((200,200))

font = Font(family=MODERN)
#print font.size
font.size=8
gc.set_font(font)

t1 = time.clock()

with gc:
    gc.set_antialias(True)
    gc.set_fill_color((0,1,0))
    gc.translate_ctm(50,50)
    gc.rotate_ctm(3.1416/4)
    gc.show_text("hello")
    gc.translate_ctm(-50,-50)
    gc.set_text_matrix(AffineMatrix())
    gc.set_fill_color((0,1,1))
    gc.show_text("hello")

t2 = time.clock()
print('antialiased:', t2 - t1)
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
