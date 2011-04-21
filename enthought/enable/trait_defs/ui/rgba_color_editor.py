from enthought.etsconfig.api import ETSConfig

if ETSConfig.toolkit == 'wx':
    from enthought.enable.traits.ui.wx.rgba_color_editor import RGBAColorEditor
elif ETSConfig.toolkit == 'qt4':
    from enthought.enable.traits.ui.qt4.rgba_color_editor import RGBAColorEditor
else:
    RGBAColorEditor = None
