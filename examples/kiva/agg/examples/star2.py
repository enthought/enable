from math import pi
from enthought.kiva import agg
   
def add_star(gc):
    gc.begin_path()
    gc.move_to(-20,-30)
    gc.line_to(0,30)
    gc.line_to(20,-30)   
    gc.line_to(-30,10)
    gc.line_to(30,10)
    gc.close_path()
    gc.move_to(-10,30)
    gc.line_to(10,30)

gc = agg.GraphicsContextArray((500,500))

gc.set_alpha(0.3)
gc.set_stroke_color((1.0,0.0,0.0))
gc.set_fill_color((0.0,1.0,0.0))

for i in range(0,600,5):
    gc.save_state()
    gc.translate_ctm(i,i)
    gc.rotate_ctm(i*pi/180.)
    add_star(gc)
    gc.draw_path()
    gc.restore_state()

gc.set_fill_color((0.5,0.5,0.5))
gc.rect(150,150,200,200)
gc.fill_path()
gc.save("star2.bmp")
