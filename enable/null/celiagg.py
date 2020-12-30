from kiva.celiagg import CompiledPath, GraphicsContext  # noqa


class NativeScrollBar(object):
    pass


class Window(object):
    pass


def font_metrics_provider():
    from kiva.fonttools import Font
    gc = GraphicsContext((1, 1))
    gc.set_font(Font())
    return gc
