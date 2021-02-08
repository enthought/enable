# (C) Copyright 2005-2021 Enthought, Inc., Austin, TX
# All rights reserved.
#
# This software is provided without warranty under the terms of the BSD
# license included in LICENSE.txt and may be redistributed only under
# the conditions described in the aforementioned license. The license
# is also available online at http://www.enthought.com/licenses/BSD.txt
#
# Thanks for using Enthought open source!
from unittest.mock import Mock

from kiva.image import GraphicsContext


class KivaTestAssistant(object):
    """ Mixin test helper for kiva drawing tests.
    """

    def create_mock_gc(self, width, height, methods=()):
        """ Create an image graphics context that with mocked methods.

        Parameters
        ----------
        width, height :
            The size of the graphics context canvas.

        methods : iterable
           the methods which are going to be mocked with a Mock object.
        """
        gc = GraphicsContext((int(width), int(height)))
        gc.clear((0.0, 0.0, 0.0, 0.0))
        for method in methods:
            setattr(gc, method, Mock())
        return gc

    def assertPathsAreProcessed(self, drawable, width=200, height=200):
        """ Check that all the paths have been compiled and processed.

        Parameters
        ----------
        drawable :
            A drawable object that has a draw method.

        width : int, optional
            The width of the array buffer (default is 200).

        height : int, optional
            The height of the array buffer (default is 200).

        note ::

           A drawable that draws nothing will pass this check.
        """
        gc = GraphicsContext((width, height))
        drawable.draw(gc)
        compiled_path = gc._get_path()
        total_vertices = compiled_path.total_vertices()
        self.assertEqual(
            total_vertices,
            0,
            msg="There are {0} vertices in compiled paths {1} that "
            "have not been processed".format(total_vertices, compiled_path),
        )

    def assertPathsAreCreated(self, drawable, width=200, height=200):
        """ Check that drawing creates paths.

        When paths and lines creation methods are used from a graphics
        context the drawing paths are compiled and processed. By using
        a mock graphics context we can check if something has been drawn.

        Parameters
        ----------
        drawable :
            A drawable object that has a draw method.

        width : int, optional
            The width of the array buffer (default is 200).

        height : int, optional
            The height of the array buffer (default is 200).
        """
        gc = self.create_mock_gc(width, height, ("draw_path", "stroke_path"))
        drawable.draw(gc)
        compiled_path = gc._get_path()
        self.assertTrue(
            compiled_path.total_vertices() > 0,
            msg="There are no compiled paths "
            "created: {0}".format(compiled_path),
        )
