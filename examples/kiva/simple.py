from kiva import constants
from kiva.image import GraphicsContext

gc = GraphicsContext((100,100))

gc.clear()
gc.set_line_cap(constants.CAP_SQUARE)
gc.set_line_join(constants.JOIN_MITER)
gc.set_stroke_color((1,0,0))
gc.set_fill_color((0,0,1))
gc.rect(0, 0, 30, 30)
gc.draw_path()

gc.save("simple.bmp")

