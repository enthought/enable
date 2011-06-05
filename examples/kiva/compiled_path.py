# CompiledPath should always be imported from the same backend as the
# GC you are using.  In this case, we are using the image GraphicsContext
# so we can save to disk when we're done, so we grab the CompiledPath
# from there as well.
from kiva.image import GraphicsContext, CompiledPath
from kiva.constants import STROKE

star_points = [(-20,-30),
               (0, 30),
               (20,-30),
               (-30,10),
               (30,10),
               (-20,-30)]

path = CompiledPath()
path.move_to(*star_points[0])
for pt in star_points[1:]:
    path.line_to(*pt)

locs = [(100,100), (100,300), (100,500), (200,100), (200,300), (200,500)]

gc = GraphicsContext((300,600))
gc.set_stroke_color((0,0,1,1))
gc.draw_path_at_points(locs, path, STROKE)
gc.save("compiled_path.png")
