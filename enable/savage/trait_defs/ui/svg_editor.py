""" Traits UI 'display only' SVG editor.
"""

#-------------------------------------------------------------------------------
#  Imports:
#-------------------------------------------------------------------------------

from enable.savage.trait_defs.ui.toolkit import toolkit_object

from traits.api import Property

from traitsui.basic_editor_factory import BasicEditorFactory

#-------------------------------------------------------------------------------
#  'SVGEditor' editor factory class:
#-------------------------------------------------------------------------------

class SVGEditor(BasicEditorFactory):

    # The editor class to be created:
    klass = Property

    def _get_klass(self):
        """ Returns the toolkit-specific editor class to be instantiated.
        """
        return toolkit_object('svg_editor:SVGEditor')
