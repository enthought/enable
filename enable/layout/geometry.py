# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!

# -----------------------------------------------------------------------------
# Rect
# -----------------------------------------------------------------------------
class BaseRect(tuple):
    """ A tuple subclass representing an (x, y, width, height)
    bounding box. Subclasses should override the __new__ method
    to enforce any necessary typing.
    """
    __slots__ = ()

    def __new__(cls, x, y, width, height):
        return super(BaseRect, cls).__new__(cls, (x, y, width, height))

    def __getnewargs__(self):
        return tuple(self)

    def __repr__(self):
        template = "%s(x=%s, y=%s, width=%s, height=%s)"
        values = (self.__class__.__name__,) + self
        return template % values

    @property
    def x(self):
        """ The 'x' position component of the rect.
        """
        return self[0]

    @property
    def y(self):
        """ The 'y' position component of the rect.
        """
        return self[1]

    @property
    def width(self):
        """ The 'width' size component of the rect.
        """
        return self[2]

    @property
    def height(self):
        """ The 'height' size component of the rect.
        """
        return self[3]


class Rect(BaseRect):
    """ A BaseRect implementation for integer values.
    """
    __slots__ = ()

    def __new__(cls, x, y, width, height):
        i = int
        return super(Rect, cls).__new__(cls, i(x), i(y), i(width), i(height))

    @property
    def box(self):
        """ The equivalent Box for this rect.
        """
        x, y, width, height = self
        return Box(y, x + width, y + height, x)

    @property
    def pos(self):
        """ The position of the rect as a Pos object.
        """
        return Pos(self.x, self.y)

    @property
    def size(self):
        """ The size of the rect as a Size object.
        """
        return Size(self.width, self.height)


class RectF(BaseRect):
    """ A BaseRect implementation for floating point values.
    """
    __slots__ = ()

    def __new__(cls, x, y, width, height):
        f = float
        return super(RectF, cls).__new__(cls, f(x), f(y), f(width), f(height))

    @property
    def box(self):
        """ The equivalent Box for this rect.
        """
        x, y, width, height = self
        return BoxF(y, x + width, y + height, x)

    @property
    def pos(self):
        """ The position of the rect as a Pos object.
        """
        return PosF(self.x, self.y)

    @property
    def size(self):
        """ The size of the rect as a Size object.
        """
        return SizeF(self.width, self.height)


# -----------------------------------------------------------------------------
# Box
# -----------------------------------------------------------------------------
class BaseBox(tuple):
    """ A tuple subclass representing a (top, right, bottom, left) box.
    Subclasses should override the __new__ method to enforce any typing.
    """
    __slots__ = ()

    @staticmethod
    def coerce_type(item):
        return item

    def __new__(cls, top=None, right=None, bottom=None, left=None):
        if isinstance(top, (tuple, BaseBox)):
            return cls(*top)
        c = cls.coerce_type
        top = c(top)
        if right is None:
            right = top
        else:
            right = c(right)
        if bottom is None:
            bottom = top
        else:
            bottom = c(bottom)
        if left is None:
            left = right
        else:
            left = c(left)
        return super(BaseBox, cls).__new__(cls, (top, right, bottom, left))

    def __getnewargs__(self):
        return tuple(self)

    def __repr__(self):
        template = "%s(top=%s, right=%s, bottom=%s, left=%s)"
        values = (self.__class__.__name__,) + self
        return template % values

    @property
    def top(self):
        """ The 'top' component of the box.
        """
        return self[0]

    @property
    def right(self):
        """ The 'right' component of the box.
        """
        return self[1]

    @property
    def bottom(self):
        """ The 'bottom' component of the box.
        """
        return self[2]

    @property
    def left(self):
        """ The 'left' component of the box.
        """
        return self[3]


class Box(BaseBox):
    """ A BaseBox implementation for integer values.
    """
    __slots__ = ()

    @staticmethod
    def coerce_type(item):
        return 0 if item is None else int(item)

    @property
    def rect(self):
        """ The equivalent Rect for this box.
        """
        top, right, bottom, left = self
        return Rect(left, top, right - left, bottom - top)

    @property
    def size(self):
        """ The Size of this box.
        """
        top, right, bottom, left = self
        return Size(right - left, bottom - top)

    @property
    def pos(self):
        """ The Pos of this box.
        """
        return Pos(self.left, self.top)


class BoxF(BaseBox):
    """ A BaseBox implementation for floating point values.
    """
    __slots__ = ()

    @staticmethod
    def coerce_type(item):
        return 0.0 if item is None else float(item)

    @property
    def rect(self):
        """ The equivalent Rect for this box.
        """
        top, right, bottom, left = self
        return RectF(left, top, right - left, bottom - top)

    @property
    def size(self):
        """ The Size of this box.
        """
        top, right, bottom, left = self
        return SizeF(right - left, bottom - top)

    @property
    def pos(self):
        """ The Pos of this box.
        """
        return PosF(self.left, self.top)


# -----------------------------------------------------------------------------
# Pos
# -----------------------------------------------------------------------------
class BasePos(tuple):
    """ A tuple subclass representing a (x, y) positions. Subclasses
    should override the __new__ method to enforce any necessary typing.
    """
    __slots__ = ()

    def __new__(cls, x, y):
        return super(BasePos, cls).__new__(cls, (x, y))

    def __getnewargs__(self):
        return tuple(self)

    def __repr__(self):
        template = "%s(x=%s, y=%s)"
        values = (self.__class__.__name__,) + self
        return template % values

    @property
    def x(self):
        """ The 'x' component of the size.
        """
        return self[0]

    @property
    def y(self):
        """ The 'y' component of the size.
        """
        return self[1]


class Pos(BasePos):
    """ An implementation of BasePos for integer values.
    """
    __slots__ = ()

    def __new__(cls, x, y):
        i = int
        return super(Pos, cls).__new__(cls, i(x), i(y))


class PosF(BasePos):
    """ An implementation of BasePos of floating point values.
    """
    __slots__ = ()

    def __new__(cls, x, y):
        f = float
        return super(PosF, cls).__new__(cls, f(x), f(y))


# -----------------------------------------------------------------------------
# Size
# -----------------------------------------------------------------------------
class BaseSize(tuple):
    """ A tuple subclass representing a (width, height) size. Subclasses
    should override the __new__ method to enforce any necessary typing.
    """
    __slots__ = ()

    @staticmethod
    def coerce_type(item):
        return item

    def __new__(cls, width=None, height=None):
        if isinstance(width, (tuple, BaseSize)):
            return cls(*width)
        c = cls.coerce_type
        width = c(width)
        if height is None:
            height = width
        else:
            height = c(height)
        return super(BaseSize, cls).__new__(cls, (width, height))

    def __getnewargs__(self):
        return tuple(self)

    def __repr__(self):
        template = "%s(width=%s, height=%s)"
        values = (self.__class__.__name__,) + self
        return template % values

    @property
    def width(self):
        """ The 'width' component of the size.
        """
        return self[0]

    @property
    def height(self):
        """ The 'height' component of the size.
        """
        return self[1]


class Size(BaseSize):
    """ A BaseSize implementation for integer values.
    """
    __slots__ = ()

    @staticmethod
    def coerce_type(item):
        return 0 if item is None else int(item)


class SizeF(BaseSize):
    """ A BaseSize implementation for floating point values.
    """
    __slots__ = ()

    @staticmethod
    def coerce_type(item):
        return 0.0 if item is None else float(item)
