
from kiva.gl import GraphicsContext
from .graphics_context import GraphicsContextEnable

class GLGraphicsContextEnable(GraphicsContextEnable, GraphicsContext):
    """ This class just binds the GraphicsContextEnable to a Kiva
    GL graphics context.
    """
    pass
