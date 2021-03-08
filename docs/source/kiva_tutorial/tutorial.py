from math import tau

import numpy as np

from kiva.api import CAP_ROUND, CIRCLE_MARKER, Font, STROKE
from kiva.image import GraphicsContext, CompiledPath

gc = GraphicsContext((600, 300))

# step 1) draw wires
gc.rect(50, 50, 500, 100)
gc.rect(200, 150, 200, 50)
gc.rect(200, 200, 200, 50)
gc.stroke_path()
gc.save("images/step_1.png")

# step 2) draw dots for wire connections
points = np.array([
    [200., 150.],
    [200., 200.],
    [400., 150.],
    [400., 200.],
    [550., 130.] 
])
gc.draw_marker_at_points(points, 4.0, CIRCLE_MARKER)
gc.save("images/step_2.png")

# step 3) Ammeter and Voltmeter
font = Font('Times New Roman', size=20)
gc.set_font(font)
with gc:  # Voltmeter
    gc.translate_ctm(50, 100)
    gc.set_fill_color((.9, .9, 0.5, 1.0))
    gc.set_line_width(3)
    gc.arc(0, 0, 20, 0.0, tau)
    gc.draw_path()

    gc.set_fill_color((0., 0., 0., 1.0))
    x, y, w, h = gc.get_text_extent('A')
    gc.show_text_at_point('A', -w/2, -h/2)

with gc:  # Ammeter
    gc.translate_ctm(300, 250)
    gc.set_fill_color((0.5, .9, 0.5, 1.0))
    gc.set_line_width(3)
    gc.arc(0, 0, 20, 0.0, tau)
    gc.draw_path()

    gc.set_fill_color((0., 0., 0., 1.0))
    x, y, w, h = gc.get_text_extent('V')
    gc.show_text_at_point('V', -w/2, -h/2)
gc.save("images/step_3.png")

# step 5) clear some space for the resistors
clear_resistor_path = CompiledPath()
clear_resistor_path.move_to(0,0)
clear_resistor_path.line_to(80, 0)

resistor_locations = [
    (150, 50),
    (350, 50),
    (260, 150),
    (260, 200)
]
with gc:
    gc.set_stroke_color((1., 1., 1., 1.))
    gc.set_line_width(2)
    gc.draw_path_at_points(resistor_locations, clear_resistor_path, STROKE)

#step 4) resistors
resistor_path = CompiledPath()
resistor_path.move_to(0,0)
resistor_path_points = [(i*10+5, 10*(-1)**i) for i in range(8)]
for x, y in resistor_path_points:
    resistor_path.line_to(x,y)
resistor_path.line_to(80, 0)
gc.draw_path_at_points(resistor_locations, resistor_path, STROKE)
gc.save("images/step_45.png")

# step 6) switch
# white out the wire 
with gc:
    gc.translate_ctm(550, 130)
    # wire connection dot markers have size 4 and we don't want to clear that
    gc.move_to(0, -4)
    gc.set_stroke_color((1., 1., 1., 1.))
    gc.set_line_width(2)
    gc.line_to(0, -30)
    gc.stroke_path()
# draw the switch
with gc:
    # move to the connected side of the switch and rotate coordinates
    # to the angle we want to draw the switch
    gc.translate_ctm(550, 100)
    gc.rotate_ctm(tau/6)
    gc.move_to(0, 0)
    gc.line_to(30, 0)
    gc.stroke_path()
gc.save("images/step_6.png")

# step 7) battery
with gc:
    gc.translate_ctm(550, 90)
    # wire connection dot markers have size 4 and we don't want to clear that
    gc.move_to(0, 0)
    gc.set_stroke_color((1., 1., 1., 1.))
    gc.set_line_width(2)
    gc.line_to(0, -30)
    gc.stroke_path()

with gc:
    gc.translate_ctm(550, 90)
    gc.move_to(0, 0)
    thin_starts = [
        (-20, 0),
        (-20, -18)
    ]
    thin_ends = [
        (20,0),
        (20, -18),
    ]
    gc.line_set(thin_starts, thin_ends)
    gc.stroke_path()
    thick_starts = [
        (-8, -10),
        (-8, -28)
    ]
    thick_ends = [
        (8, -10),
        (8, -28),
    ]
    gc.set_line_width(8)
    gc.set_line_cap(CAP_ROUND)
    gc.line_set(thick_starts, thick_ends)
    gc.stroke_path()

gc.save("images/tutorial.png")
