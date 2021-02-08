# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
from math import acos, atan2

from numpy import array, dot, pi, sin, sqrt


def two_point_arc_to_kiva_arc(p1, p2, theta):
    """
    Converts an arc in two point and subtended angle  format (startpoint,
    endpoint, theta (positive for ccw, negative for cw)) into kiva format
    (x, y, radius, start_angle, end_angle, cw)
    """

    chord = p2 - p1
    chordlen = sqrt(dot(chord, chord))
    radius = abs(chordlen / (2 * sin(theta / 2)))
    altitude = sqrt(pow(radius, 2) - pow(chordlen / 2, 2))
    if theta > pi or theta < 0:
        altitude = -altitude
    chordmidpoint = (p1 + p2) / 2
    rotate90 = array(((0.0, -1.0), (1.0, 0.0)))
    centerpoint = dot(rotate90, (chord / chordlen)) * altitude + chordmidpoint

    start_angle = atan2(*(p1 - centerpoint)[::-1])
    end_angle = start_angle + theta
    if theta < 0:
        start_angle, end_angle, = end_angle, start_angle
    cw = False
    radius = abs(radius)
    return (centerpoint[0], centerpoint[1], radius, start_angle, end_angle, cw)


def arc_to_tangent_points(start, p1, p2, radius):
    """ Given a starting point, two endpoints of a line segment, and a radius,
        calculate the tangent points for arc_to().
    """

    def normalize_vector(x, y):
        """ Given a vector, return its unit length representation.
        """
        length = sqrt(x ** 2 + y ** 2)
        if length <= 1e-6:
            return (0.0, 0.0)
        return (x / length, y / length)

    # calculate the angle between the two line segments
    v1 = normalize_vector(start[0] - p1[0], start[1] - p1[1])
    v2 = normalize_vector(p2[0] - p1[0], p2[1] - p1[1])
    angle = acos(v1[0] * v2[0] + v1[1] * v2[1])

    # punt if the half angle is zero or a multiple of pi
    sin_half_angle = sin(angle / 2.0)
    if sin_half_angle == 0.0:
        return (p1, p2)

    # calculate the distance from p1 to the center of the arc
    dist_to_center = radius / sin_half_angle
    # calculate the distance from p1 to each tangent point
    dist_to_tangent = sqrt(dist_to_center ** 2 - radius ** 2)

    # calculate the tangent points
    t1 = (p1[0] + v1[0] * dist_to_tangent, p1[1] + v1[1] * dist_to_tangent)
    t2 = (p1[0] + v2[0] * dist_to_tangent, p1[1] + v2[1] * dist_to_tangent)

    return (t1, t2)
