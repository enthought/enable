

from scipy import pi
from kiva.image import GraphicsContext

def draw_ellipse(gc, x, y, major, minor, angle):
    """ Draws an ellipse given major and minor axis lengths.  **angle** is
    the angle between the major axis and the X axis, in radians.
    """
    with gc:
        gc.translate_ctm(x, y)
        ratio = float(major) / minor
        gc.rotate_ctm(angle)
        gc.scale_ctm(ratio, 1.0)
        gc.arc(0, 0, minor, 0.0, 2 * pi)
        gc.stroke_path()
        gc.move_to(-minor, 0)
        gc.line_to(minor, 0)
        gc.move_to(0, -minor)
        gc.line_to(0, minor)
        gc.stroke_path()

gc = GraphicsContext((300,300))

gc.set_alpha(0.3)
gc.set_stroke_color((1.0,0.0,0.0))
gc.set_fill_color((0.0,1.0,0.0))
gc.rect(95, 95, 10, 10)
gc.fill_path()
draw_ellipse(gc, 100, 100, 35.0, 25.0, pi / 6)
gc.save("ellipse.bmp")


