import sys
from kiva import agg

gc = agg.GraphicsContextArray((100, 100))
gc.move_to(0, 0)
gc.line_to(100, 100)
gc.stroke_path()

gc.save("bob.bmp")
gc.save("bob.jpg")

gc.convert_pixel_format("rgb24")
gc.save("bob1.bmp")

if sys.platform == "win32":
    from kiva.agg import GraphicsContextSystem
    gc = GraphicsContextSystem((100, 100))
    gc.move_to(0, 0)
    gc.line_to(100, 100)
    gc.stroke_path()
    gc.save("bob2.bmp")
    gc.convert_pixel_format("rgb24")
    gc.save("bob3.bmp")
