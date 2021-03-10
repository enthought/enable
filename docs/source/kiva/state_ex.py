import math
from kiva import CAP_ROUND, CAP_SQUARE, JOIN_ROUND
from kiva.image import GraphicsContext

gc = GraphicsContext((600, 600))

gc.scale_ctm(2, 2)
gc.translate_ctm(150, 150)

gc.set_stroke_color((0.66, 0.88, 0.66))
gc.set_line_width(7.0)
gc.set_line_join(JOIN_ROUND)
gc.set_line_cap(CAP_SQUARE)

for i in range(0, 12):
    theta = i*2*math.pi / 12.0
    with gc:
        gc.rotate_ctm(theta)
        gc.translate_ctm(105, 0)
        gc.set_stroke_color((1 - (i / 12), math.fmod(i / 6, 1), i / 12))
        gc.set_line_width(10.0)
        gc.set_line_cap(CAP_ROUND)
        gc.rect(0, 0, 25, 25)
        gc.stroke_path()

    with gc:
        gc.rotate_ctm(theta)
        gc.translate_ctm(20, 0)
        gc.move_to(0, 0)
        gc.line_to(80, 0)
        gc.stroke_path()

gc.save("state_ex.png")
