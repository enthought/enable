from traits.etsconfig.api import ETSConfig

if ETSConfig.toolkit == 'wx':
    from .wx.rgba_color_editor import RGBAColorEditor
elif ETSConfig.toolkit == 'qt4':
    from .qt4.rgba_color_editor import RGBAColorEditor
else:
    RGBAColorEditor = None
