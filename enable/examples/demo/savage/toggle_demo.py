import os

from enable.savage.trait_defs.ui.svg_button import SVGButton
from traits.api import HasTraits
from traitsui.api import Item, View

ROOT = os.path.dirname(__file__)
PAUSE_ICON = os.path.join(ROOT, 'player_pause.svg')
RESUME_ICON = os.path.join(ROOT, 'player_play.svg')


class SVGDemo(HasTraits):
    pause = SVGButton('Pause', filename=PAUSE_ICON,
                      toggle_filename=RESUME_ICON,
                      toggle_state=True,
                      toggle_label='Resume',
                      toggle_tooltip='Press button to "Resume".',
                      tooltip='Press button to "Pause".',
                      toggle=True)

    trait_view = View(Item('pause'))


demo = SVGDemo()

if __name__ == "__main__":
    demo.configure_traits()
