from math import pi
from kiva.image import GraphicsContext

gc = GraphicsContext((600, 600))

path = gc.get_empty_path()
path.move_to(10, 40)
path.line_to(60, 40)
path.line_to(60, 90)
path.close_path()

gc.scale_ctm(2, 2)
gc.translate_ctm(150, 150)
for i in range(0, 12):
    gc.rotate_ctm(2*pi / 12.0)
    gc.set_fill_color((i / 12.0, 0.0, 1.0 - (i / 12.0)))
    gc.add_path(path)
    gc.fill_path()

gc.save("compiled_path_ex.png")
