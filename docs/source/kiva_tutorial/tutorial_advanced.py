from math import tau

import numpy as np

from kiva.api import CAP_ROUND, CIRCLE_MARKER, FILL, Font, STROKE
from kiva.image import GraphicsContext, CompiledPath


def draw_wire_with_components(gc, start, end, component_locations):
    """
    Draws a straight, axis aligned, wire with gaps in it for components. This
    function assumes the component locations are in order they are encountered
    when moving from start to end. 
    
    Parameters
    ----------
    gc : GraphicsContext
        The Graphics context doing the drawing
    start : 2-tuple
        The start point of the wire
    end : 2-tuple
        The end point of the wire
    component_locations : List of pairs of 2-tuple
        The start and end points of the components in order encountered
    """
    with gc:
        gc.set_stroke_color((0., 0., 0., 1.))
        gc.set_line_width(1.0)

        gc.move_to(*start)
        for comp_start, comp_end in component_locations:
            gc.line_to(*comp_start)
            gc.move_to(*comp_end)
        gc.line_to(*end)
        gc.stroke_path()


def draw_rect_wire_frame_with_components(gc, x, y, w, h, component_locations):
    """
    Draws an axis aligned rectangle of wire with gaps in it for components.
    component_locations is a list of pairs of points corresponding to the start
    and end of a component.  The function assumes the component locations are
    in order they are encountered when moving clockwise around the rectangle
    starting at the lower left corner.

    Parameters
    ----------
    gc : GraphicsContext
        The Graphics context doing the drawing
    x : int
        The left X coordinate of the rectangle
    y : int
        The bottom Y coordinate of the rectangle
    w : int
        The width of the rectangle
    h : int
        The height of the rectangle
    component_locations : List of pairs of 2-tuple
        The start and end points of the components in order encountered
    """
    left_comps = [
        comp_loc for comp_loc in component_locations if comp_loc[0][0] == x
    ]
    top_comps = [
        comp_loc for comp_loc in component_locations if comp_loc[0][1] == y + h
    ]
    right_comps =[
        comp_loc for comp_loc in component_locations if comp_loc[0][0] == x + w
    ]
    bottom_comps = [
        comp_loc for comp_loc in component_locations if comp_loc[0][1] == y
    ]

    draw_wire_with_components(gc, (x, y), (x, y + h), left_comps)
    draw_wire_with_components(gc, (x, y + h), (x + w, y + h), top_comps)
    draw_wire_with_components(gc, (x + w, y + h), (x + w, y), right_comps)
    draw_wire_with_components(gc, (x + w, y), (x, y), bottom_comps)


def draw_wire_connections_at_points(gc, points):
    """
    Draw wire connections at each of the given points. This function checks if
    the graphics context implements optimized methods for doing so, and draws
    using the most optimal approach available.

    Parameters
    ----------
    gc : GraphicsContext
        The Graphics context doing the drawing
    points : List of pairs of 2-tuple
        The points where wire connections are to be drawn
    """

    if hasattr(gc, 'draw_marker_at_points'):
        gc.draw_marker_at_points(points, 4.0, CIRCLE_MARKER)
    
    else:
        wire_connection_path = CompiledPath()
        wire_connection_path.move_to(0,0)
        wire_connection_path.arc(0, 0, 4, 0, tau)

        if hasattr(gc, 'draw_path_at_points'):
            gc.draw_path_at_points(points, wire_connection_path, FILL)
        else:
            for point in points:
                with gc:
                    gc.translate_ctm(point[0], point[1])
                    gc.add_path(wire_connection_path)
                    gc.fill_path()


def create_resistor_path():
    """
    Creates a CompiledPath for a resistor which can then be re-used as needed.

    Returns
    -------
    CompiledPath
        The reistor compiled path
    """
    resistor_path = CompiledPath()
    resistor_path.move_to(0,0)
    resistor_path_points = [(i*10+5, 10*(-1)**i) for i in range(8)]
    for x, y in resistor_path_points:
        resistor_path.line_to(x,y)
    resistor_path.line_to(80, 0)

    return resistor_path


def draw_resistors_at_points(gc, points, resistor_path):
    """
    Draw a resistor at each of the given points. This function checks if
    the graphics context implements an optimized method for doing so, and draws
    using the most optimal approach available.

    Parameters
    ----------
    gc : GraphicsContext
        The graphics context doing the drawing.
    points : List of pairs of 2-tuple
        The points where resistors are to be drawn 
    resistor_path : CompiledPath
        The resistor path we wish to draw
    """

    if hasattr(gc, 'draw_path_at_points'):
        gc.draw_path_at_points(points, resistor_path, STROKE)
    else:
        for point in points:
            with gc:
                gc.translate_ctm(point[0], point[1])
                gc.add_path(resistor_path)
                gc.stroke_path()


def draw_meter(gc, location, color, text):
    """
    Draws a meter of the given color, with the given text, at the given
    location.

    Parameters
    ----------
    gc : GraphicsContext
        The graphics context doing the drawing.
    location : 2-tuple
        The point where the meter is to be drawn 
    color : 3 or 4 component tuple (R, G, B[, A])
        The color of the meter
    text : str
        The text to be placed in the center of the meter symbol
    """
    font = Font('Times New Roman', size=20)
    with gc:
        gc.set_font(font)
        gc.set_fill_color(color)
        gc.set_line_width(3)
        gc.translate_ctm(*location)

        gc.arc(0, 0, 20, 0.0, tau)
        gc.draw_path()

        gc.set_fill_color((0., 0., 0., 1.0))
        x, y, w, h = gc.get_text_extent(text)
        gc.show_text_at_point(text, -w/2, -h/2)


def draw_switch(gc, location, angle):
    """
    Draws a switch at given location.  Assumes location is the connected side
    of the switch, and angle assumes orientation facing directly accross the
    switch.

    Parameters
    ----------
    gc : GraphicsContext
        The graphics context doing the drawing.
    location : 2-tuple
        The point where the switch is to be drawn 
    angle : float
        The angle of the switch
    """
    with gc:
        gc.translate_ctm(*location)
        gc.rotate_ctm(angle)
        gc.move_to(0, 0)
        gc.line_to(30, 0)
        gc.stroke_path()


def draw_battery(gc, location):
    """
    Draws a battery at given location. Battery will extend down from the given
    location.

    Parameters
    ----------
    gc : GraphicsContext
        The graphics context doing the drawing.
    location : 2-tuple
        The point where the switch is to be drawn 
    """
    with gc:
        gc.translate_ctm(*location)
        gc.move_to(0, 0)
        thin_starts = [(-20, 0), (-20, -18)]
        thin_ends = [(20,0), (20, -18)]
        gc.line_set(thin_starts, thin_ends)
        gc.stroke_path()
        thick_starts = [(-8, -10), (-8, -28)]
        thick_ends = [(8, -10), (8, -28)]
        gc.set_line_width(8)
        gc.set_line_cap(CAP_ROUND)
        gc.line_set(thick_starts, thick_ends)
        gc.stroke_path()


if __name__ == "__main__":

    gc = GraphicsContext((600, 300))

    # step 1) Draw a skeleton of the circuit
    component_locations = [
        ((260, 150), (340, 150)),
        ((550, 130), (550, 100)),
        ((550, 90), (550, 60)),
        ((430, 50), (350, 50)),
        ((230, 50), (150, 50))
    ]
    draw_rect_wire_frame_with_components(
        gc, 50, 50, 500, 100, component_locations
    )
    draw_rect_wire_frame_with_components(
        gc, 200, 200, 200, 50, [((340, 200), (260, 200))]
    )
    draw_wire_with_components(gc, (200, 150), (200, 200), [])
    draw_wire_with_components(gc, (400, 150), (400, 200), [])


    # step 2) draw dots for wire connections
    points = [(200, 150), (200, 200), (400, 150), (400, 200), (550, 130)]
    draw_wire_connections_at_points(gc, points)

    # step 3) Draw the meters
    draw_meter(gc, (50, 100), (.9, .9, 0.5, 1.0), 'A')  # Ammeter
    draw_meter(gc, (300, 250), (0.5, .9, 0.5, 1.0), 'V')  # Voltmeter

    #step 4) Draw the resistors
    resistor_path = create_resistor_path()
    resistor_locations = [(150, 50), (350, 50), (260, 150), (260, 200)]
    draw_resistors_at_points(gc, resistor_locations, resistor_path)

    # step 6) Draw the switch
    draw_switch(gc, (550,100), tau/6)

    # step 7) Draw the battery
    draw_battery(gc, (550,90))
    
    gc.save("images/tutorial_advanced.png")
