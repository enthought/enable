# Copyright (c) 2008-2013 by Enthought, Inc.
# All rights reserved.
from mock import Mock

from kiva.image import GraphicsContext


class KivaTestAssistant(object):
    """ Mixin test helper for kiva drawing tests.

    """

    def create_mock_gc(
            self, width, height, stroke_path=None, draw_path=None):
        """ Create an image graphics context that will mock the stroke_path
        and draw_path methods.

        Parameters
        ----------
        width, height :
            The size of the graphics context canvas.

        stroke_path : callable, optional
            A callable to use as the stroke_path method (default is Mock()).

        draw_path : callable, optional
            A callable to use as the draw_path method (default is Mock()).


        """
        gc = GraphicsContext((width, height))
        gc.clear((0.0, 0.0, 0.0, 0.0))
        gc.stroke_path = Mock() if stroke_path is None else stroke_path
        gc.draw_path = Mock() if draw_path is None else draw_path
        return gc

    def assertPathsAreProcessed(self, drawable, width=200, height=200):
        """ Check that drawing does not leave paths unused in the GC cache.

        When drawing paths are created

        Parameters
        ----------
        drawable :
            A drawable object that has a draw method.

        width : int, optional
            The width of the array buffer

        height : int, optional
            The height of the array buffer

        """
        gc = GraphicsContext((width, height))
        drawable.draw(gc)
        compiled_path = gc._get_path()
        self.assertEqual(
            compiled_path.total_vertices(), 0,
            msg='There are compiled paths that '
            'have not been processed: {0}'.format(compiled_path))

    def assertPathsAreCreated(self, drawable, width=200, height=200):
        """ Check that drawing creates paths.



        Parameters
        ----------
        drawable :
            A drawable object that has a draw method.

        width : int, optional
            The width of the array buffer (default is 200)

        height : int, optional
            The height of the array buffer (default is 200)

        """
        gc = self.create_mock_gc(width, height)
        drawable.draw(gc)
        compiled_path = gc._get_path()
        self.assertGreater(
            compiled_path.total_vertices(), 0,
            msg='There are no compiled paths '
            'created: {0}'.format(compiled_path))
