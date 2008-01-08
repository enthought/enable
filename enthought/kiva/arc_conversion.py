from numpy import sqrt, dot, sin, dot, array, pi

from math import atan2

def two_point_arc_to_kiva_arc(p1, p2, theta):
    """
    Converts an arc in two point and subtended angle  format (startpoint,
    endpoint, theta (positive for ccw, negative for cw)) into kiva format
    (x, y, radius, start_angle, end_angle, cw)
    """

    chord = p2-p1
    chordlen = sqrt(dot(chord, chord))
    radius = abs(chordlen/(2*sin(theta/2)))
    altitude = sqrt(pow(radius, 2)-pow(chordlen/2, 2))
    if theta>pi or theta<0:
        altitude = -altitude
    chordmidpoint = (p1+p2)/2
    rotate90 = array(((0.0,-1.0),
                      (1.0, 0.0)))
    centerpoint = dot(rotate90, (chord/chordlen))*altitude + chordmidpoint

    start_angle = atan2(*(p1-centerpoint)[::-1])
    end_angle = start_angle+theta
    if theta<0:
        start_angle, end_angle, = end_angle, start_angle
    cw = False
    radius = abs(radius)
    return (centerpoint[0], centerpoint[1], radius, start_angle, end_angle, cw)

