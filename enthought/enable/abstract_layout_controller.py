
from traits.api import HasTraits

class AbstractLayoutController(HasTraits):

    def layout(self, component):
        raise NotImplementedError

