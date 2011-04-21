import os

from traits.api import HasTraits
from traitsui.api import View, Item
from enable.savage.trait_defs.ui.svg_button import SVGButton


pause_icon = os.path.join(os.path.dirname(__file__), 'player_pause.svg')
resume_icon = os.path.join(os.path.dirname(__file__), 'player_play.svg')

class SVGDemo(HasTraits):

    pause = SVGButton('Pause', filename=pause_icon,
                      toggle_filename=resume_icon,
                      toggle_state=True,
                      toggle_label='Resume',
                      toggle_tooltip='Resume',
                      tooltip='Pause', toggle=True)

    trait_view = View(Item('pause'))

SVGDemo().configure_traits()
