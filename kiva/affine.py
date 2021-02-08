# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
""" Functions for affine matrices.

    :Copyright:   Space Telescope Science Institute
    :License:     BSD Style
    :Author:      Eric Jones, Enthought, Inc., eric@enthought.com

    These affine operations are based coordinate system transformations,
    not translations of pts, etc.  To translate a point, you multiply
    it by the affine transform matrix::

                         a    b    0
            [x, y, 1] *  c    d    0 = [x', y', 1]
                         tx   ty   1

    This is the opposite order of multiplication from what many are
    accustomed to.

    Here is a useful link:

        http://mathworld.wolfram.com/AffineTransformation.html

    Notes:
        I'm not using a class because of possible speed implications.
        Currently the affine transform is a 3x3 array.  Other tools use
        a 6-tuple of (a, b, c, d, tx, ty) to represent the transform because
        the other 3 array entries are constant.  Other code should call
        methods from this module instead of manipulating the array,
        in case the implementation is changed at some future date.
"""

from numpy import (
    arctan2, array, array_equal, cos, dot, eye, float64, ones, sin, zeros
)


# -----------------------------------------------------------------------------
# Affine transform construction
# -----------------------------------------------------------------------------


def affine_identity():
    """ Returns a new identity affine_transform object.
    """
    return eye(3, 3)


def affine_from_values(a, b, c, d, tx, ty):
    """ Return the affine matrix corresponding to the values

    The result is the array::
            [ a    b    0 ]
            [ c    d    0 ]
            [ tx   ty   1 ]

    """
    transform = array(((a, b, 0), (c, d, 0), (tx, ty, 1)), float64)
    return transform


def affine_from_scale(sx, sy):
    """ Returns an affine transform providing the given scaling.
    """
    r = affine_identity()
    return scale(r, sx, sy)


def affine_from_rotation(angle):
    """ Returns an affine transform rotated by angle in radians.
    """
    r = affine_identity()
    return rotate(r, angle)


def affine_from_translation(x, y):
    """ Returns an affine transform with the given translation.
    """
    r = affine_identity()
    return translate(r, x, y)


# -----------------------------------------------------------------------------
# Affine transform manipulation
# -----------------------------------------------------------------------------


def scale(transform, sx, sy):
    """ Returns a scaled version of the transform by the given values.

    Scaling is done using the following formula::

            sx  0  0     a  b  0      sx*a sx*b  0
            0  sy  0  *  c  d  0  =   sy*c sy*d  0
            0   0  1    tx ty  1       0    0    1

    """
    # this isn't the operation described above, but produces the
    # same results.
    scaled = transform.copy()
    scaled[0] *= sx
    scaled[1] *= sy
    return scaled


def rotate(transform, angle):
    """ Rotates transform by angle in radians.

    Rotation is done using the following formula::

          cos(x)  sin(x)  0       a  b   0
         -sin(x)  cos(x)  0   *   c  d   0 =
            0       0     1      tx ty   1

    ::

                 cos(x)*a+sin(x)*b   cos(x)*b+sin(x)*d    0
                -sin(x)*a+cos(x)*c  -sin(x)*b+cos(x)*d    0
                         tx                  ty           1

    where x = angle.
    """
    a = cos(angle)
    b = sin(angle)
    c = -b
    d = a
    tx = 0.0
    ty = 0.0
    rot = affine_from_values(a, b, c, d, tx, ty)
    return dot(rot, transform)


def translate(transform, x, y):
    """ Returns transform translated by (x, y).

        Translation::

            1  0  0      a   b   0         a          b        0
            0  1  0   *  c   d   0  =      c          d        0
            x  y  1     tx  ty   1     x*a+y*c+y  x*b+y*d+ty   1
    """
    r = affine_identity()
    r[2, 0] = x
    r[2, 1] = y
    return dot(r, transform)


def concat(transform, other):
    """ Returns the concatenation of transform with other.  This
        is simply transform pre-multiplied by other.
    """
    return dot(other, transform)


def invert(m):
    """ Returns the inverse of the transform, m.
    """
    inv = zeros(m.shape, float64)
    det = m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0]

    inv[0, 0] = m[1, 1]
    inv[0, 1] = -m[0, 1]
    inv[0, 2] = 0

    inv[1, 0] = -m[1, 0]
    inv[1, 1] = m[0, 0]
    inv[1, 2] = 0

    inv[2, 0] = m[1, 0] * m[2, 1] - m[1, 1] * m[2, 0]
    inv[2, 1] = -m[0, 0] * m[2, 1] + m[0, 1] * m[2, 0]
    inv[2, 2] = m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0]
    inv /= det
    return inv


# -----------------------------------------------------------------------------
# Affine transform information
# -----------------------------------------------------------------------------

IDENTITY = affine_identity()


def is_identity(m):
    """ Tests whether an affine transform is the identity transform.
    """
    return array_equal(m, IDENTITY)


def affine_params(m):
    """ Returns the a, b, c, d, tx, ty values of an
        affine transform.
    """
    a = m[0, 0]
    b = m[0, 1]
    c = m[1, 0]
    d = m[1, 1]
    tx = m[2, 0]
    ty = m[2, 1]
    return a, b, c, d, tx, ty


def tsr_factor(m):
    """ Factors a matrix as if it is the product of translate/scale/rotate
        matrices applied (i.e., concatenated) in that order.  It returns:

            tx, ty, sx, sy, angle

        where tx and ty are the translations, sx and sy are the scaling
        values and angle is the rotational angle in radians.

        If the input matrix was created in a way other than concatenating
        t/s/r matrices, the results could be wrong.  For example, if there
        is any skew in the matrix, the returned results are wrong.

        Needs Test!
    """

    # -------------------------------------------------------------------------
    # Extract Values from Matrix
    #
    # Translation values are correct as extracted.  Rotation and
    # scaling need a little massaging.
    # -------------------------------------------------------------------------
    a, b, c, d, tx, ty = affine_params(m)

    # -------------------------------------------------------------------------
    # Rotation -- tan(angle) = b/d
    # -------------------------------------------------------------------------
    angle = arctan2(b, d)

    # -------------------------------------------------------------------------
    # Scaling
    #
    # sx = a/cos(angle) or sx = -c/sin(angle)
    # sy = d/cos(angle) or sy =  b/sin(angle)
    # -------------------------------------------------------------------------
    cos_ang = cos(angle)
    sin_ang = sin(angle)
    if cos_ang != 0.0:
        sx, sy = a / cos_ang, d / cos_ang
    else:
        sx, sy = -c / sin_ang, b / sin_ang

    return tx, ty, sx, sy, angle


def trs_factor(m):
    """ Factors a matrix as if it is the product of translate/rotate/scale
        matrices applied (i.e., concatenated) in that order.  It returns:

            tx,ty,sx,sy,angle

        where tx and ty are the translations, sx and sy are the scaling
        values and angle is the rotational angle in radians.

        If the input matrix was created in a way other than concatenating
        t/r/s matrices, the results could be wrong.  For example, if there
        is any skew in the matrix, the returned results are wrong.

        Needs Test!
    """

    # ------------------------------------------------------------------------
    # Extract Values from Matrix
    #
    # Translation values are correct as extracted.  Rotation and
    # scaling need a little massaging.
    # ------------------------------------------------------------------------
    a, b, c, d, tx, ty = affine_params(m)

    # ------------------------------------------------------------------------
    # Rotation -- tan(angle) = -c/d
    # ------------------------------------------------------------------------
    angle = arctan2(-c, d)

    # ------------------------------------------------------------------------
    # Scaling
    #
    # sx = a/cos(angle) or sx =  b/sin(angle)
    # sy = d/cos(angle) or sy =  -c/sin(angle)
    # ------------------------------------------------------------------------
    cos_ang = cos(angle)
    sin_ang = sin(angle)
    if cos_ang != 0.0:
        sx, sy = a / cos_ang, d / cos_ang
    else:
        sx, sy = b / sin_ang, -c / sin_ang

    return tx, ty, sx, sy, angle


# -----------------------------------------------------------------------------
# Transforming points and arrays of points
# -----------------------------------------------------------------------------


def transform_point(ctm, pt):
    """ Returns pt transformed by the affine transform, ctm.
    """
    p1 = ones(3, float64)
    p1[:2] = pt
    res = dot(p1, ctm)[:2]
    return res


def transform_points(ctm, pts):
    """ Transforms an array of points using the affine transform, ctm.
    """
    if is_identity(ctm):
        res = pts
    else:
        x = pts[..., 0]
        y = pts[..., 1]
        a, b, c, d, tx, ty = affine_params(ctm)
        res = zeros(pts.shape, float64)
        res[..., 0] = a * x + c * y + tx
        res[..., 1] = b * x + d * y + ty
    return res
