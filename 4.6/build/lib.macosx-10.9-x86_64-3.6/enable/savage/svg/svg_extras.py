# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
""" Extra math for implementing SVG on top of Kiva.
"""
from math import acos, sin, cos, hypot, ceil, sqrt, radians, degrees
import warnings


def bezier_arc(x1, y1, x2, y2, start_angle=0, extent=90):
    """ Compute a cubic Bezier approximation of an elliptical arc.

    (x1, y1) and (x2, y2) are the corners of the enclosing rectangle.  The
    coordinate system has coordinates that increase to the right and down.
    Angles, measured in degress, start with 0 to the right (the positive
    X axis) and increase counter-clockwise.  The arc extends from start_angle
    to start_angle+extent.  I.e. start_angle=0 and extent=180 yields an
    openside-down semi-circle.

    The resulting coordinates are of the form (x1,y1, x2,y2, x3,y3, x4,y4)
    such that the curve goes from (x1, y1) to (x4, y4) with (x2, y2) and
    (x3, y3) as their respective Bezier control points.
    """

    x1, y1, x2, y2 = min(x1, x2), max(y1, y2), max(x1, x2), min(y1, y2)

    if abs(extent) <= 90:
        frag_angle = float(extent)
        nfrag = 1
    else:
        nfrag = int(ceil(abs(extent) / 90.0))
        if nfrag == 0:
            warnings.warn("Invalid value for extent: %r" % extent)
            return []
        frag_angle = float(extent) / nfrag

    x_cen = (x1 + x2) / 2.0
    y_cen = (y1 + y2) / 2.0
    rx = (x2 - x1) / 2.0
    ry = (y2 - y1) / 2.0
    half_angle = radians(frag_angle) / 2
    kappa = abs(4.0 / 3.0 * (1.0 - cos(half_angle)) / sin(half_angle))

    if frag_angle < 0:
        sign = -1
    else:
        sign = 1

    point_list = []

    for i in range(nfrag):
        theta0 = radians(start_angle + i * frag_angle)
        theta1 = radians(start_angle + (i + 1) * frag_angle)
        c0 = cos(theta0)
        c1 = cos(theta1)
        s0 = sin(theta0)
        s1 = sin(theta1)
        if frag_angle > 0:
            signed_kappa = -kappa
        else:
            signed_kappa = kappa
        point_list.append(
            (
                x_cen + rx * c0,
                y_cen - ry * s0,
                x_cen + rx * (c0 + signed_kappa * s0),
                y_cen - ry * (s0 - signed_kappa * c0),
                x_cen + rx * (c1 - signed_kappa * s1),
                y_cen - ry * (s1 + signed_kappa * c1),
                x_cen + rx * c1,
                y_cen - ry * s1,
            )
        )

    return point_list


def angle(x1, y1, x2, y2):
    """ The angle in degrees between two vectors.
    """
    sign = 1.0
    usign = x1 * y2 - y1 * x2
    if usign < 0:
        sign = -1.0
    num = x1 * x2 + y1 * y2
    den = hypot(x1, y1) * hypot(x2, y2)
    ratio = min(max(num / den, -1.0), 1.0)
    return sign * degrees(acos(ratio))


def transform_from_local(xp, yp, cphi, sphi, mx, my):
    """ Transform from the local frame to absolute space.
    """
    x = xp * cphi - yp * sphi + mx
    y = xp * sphi + yp * cphi + my
    return (x, y)


def elliptical_arc_to(path, rx, ry, phi, large_arc_flag, sweep_flag, x1, y1,
                      x2, y2):
    """ Add an elliptical arc to the kiva CompiledPath by approximating it with
    Bezier curves or a line segment.

    Algorithm taken from the SVG 1.1 Implementation Notes:
        http://www.w3.org/TR/SVG/implnote.html#ArcImplementationNotes
    """
    # Basic normalization.
    rx = abs(rx)
    ry = abs(ry)
    phi = phi % 360

    # Check for certain special cases.
    if x1 == x2 and y1 == y2:
        # Omit the arc.
        # x1 and y1 can obviously remain the same for the next segment.
        return []
    if rx == 0 or ry == 0:
        # Line segment.
        path.line_to(x2, y2)
        return []

    rphi = radians(phi)
    cphi = cos(rphi)
    sphi = sin(rphi)

    # Step 1: Rotate to the local coordinates.
    dx = 0.5 * (x1 - x2)
    dy = 0.5 * (y1 - y2)
    x1p = cphi * dx + sphi * dy
    y1p = -sphi * dx + cphi * dy
    # Ensure that rx and ry are large enough to have a unique solution.
    lam = (x1p / rx) ** 2 + (y1p / ry) ** 2
    if lam > 1.0:
        scale = sqrt(lam)
        rx *= scale
        ry *= scale

    # Step 2: Solve for the center in the local coordinates.
    num = max((rx * ry) ** 2 - (rx * y1p) ** 2 - (ry * x1p) ** 2, 0.0)
    den = (rx * y1p) ** 2 + (ry * x1p) ** 2
    a = sqrt(num / den)
    cxp = a * rx * y1p / ry
    cyp = -a * ry * x1p / rx
    if large_arc_flag == sweep_flag:
        cxp = -cxp
        cyp = -cyp

    # Step 3: Transform back.
    mx = 0.5 * (x1 + x2)
    my = 0.5 * (y1 + y2)

    # Step 4: Compute the start angle and the angular extent of the arc.
    # Note that theta1 is local to the phi-rotated coordinate space.
    dx = (x1p - cxp) / rx
    dy = (y1p - cyp) / ry
    dx2 = (-x1p - cxp) / rx
    dy2 = (-y1p - cyp) / ry
    theta1 = angle(1, 0, dx, dy)
    dtheta = angle(dx, dy, dx2, dy2)
    if not sweep_flag and dtheta > 0:
        dtheta -= 360
    elif sweep_flag and dtheta < 0:
        dtheta += 360

    # Step 5: Break it apart into Bezier arcs.
    arcs = []
    control_points = bezier_arc(
        cxp - rx, cyp - ry, cxp + rx, cyp + ry, theta1, dtheta
    )
    for x1p, y1p, x2p, y2p, x3p, y3p, x4p, y4p in control_points:
        # Transform them back to asbolute space.
        args = (
            transform_from_local(x2p, y2p, cphi, sphi, mx, my)
            + transform_from_local(x3p, y3p, cphi, sphi, mx, my)
            + transform_from_local(x4p, y4p, cphi, sphi, mx, my)
        )
        arcs.append(args)

    return arcs
