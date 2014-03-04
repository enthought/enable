from enable.gadgets.ctf.piecewise import PiecewiseFunction
from enable.gadgets.ctf.utils import (
    FunctionUIAdapter, AlphaFunctionUIAdapter, ColorFunctionUIAdapter,
    clip_to_unit
)
from enable.tools.api import ValueDragTool
from traits.api import Event, Instance, Tuple, Type


class ValueMapper(object):
    """ A simple mapper for ValueDragTool objects.
    """
    def __init__(self, obj, attr_name):
        self.obj = obj
        self.attr_name = attr_name

    def map_data(self, screen):
        size = getattr(self.obj, self.attr_name)
        return clip_to_unit(screen / size)

    def map_screen(self, data):
        size = getattr(self.obj, self.attr_name)
        return clip_to_unit(data) * size


class FunctionEditorTool(ValueDragTool):
    """ A value drag tool for editing a PiecewiseFunction.
    """

    # The function being edited
    function = Instance(PiecewiseFunction)

    # An event to trigger when the function is updated
    function_updated = Event

    # A tuple containing the index and starting value of the item being edited
    edit_value = Tuple

    # A factory for the FunctionUIAdapter to use
    ui_adapter_klass = Type

    # The helper object for screen <=> function translation
    _ui_adapter = Instance(FunctionUIAdapter)

    #------------------------------------------------------------------------
    # Traits handlers
    #------------------------------------------------------------------------

    def _x_mapper_default(self):
        return ValueMapper(self.component, 'width')

    def _y_mapper_default(self):
        return ValueMapper(self.component, 'height')

    def __ui_adapter_default(self):
        return self.ui_adapter_klass(component=self.component,
                                     function=self.function)

    #------------------------------------------------------------------------
    # ValueDragTool methods
    #------------------------------------------------------------------------

    def is_draggable(self, x, y):
        """ Returns whether the (x,y) position is in a region that is OK to
        drag.

        Used by the tool to determine when to start a drag.
        """
        index = self._ui_adapter.function_index_at_position(x, y)
        if index is not None:
            self.edit_value = (index, self.function.value_at(index))
            return True

        self.edit_value = tuple()
        return False

    def get_value(self):
        """ Return the current value that is being modified. """
        return self.edit_value


class AlphaFunctionEditorTool(FunctionEditorTool):
    """ A FuctionEditorTool for an opacity function.
    """
    ui_adapter_klass = AlphaFunctionUIAdapter

    def set_delta(self, value, delta_x, delta_y):
        """ Set the value that is being modified
        """
        index, start_value = value
        if index == 0:
            x_limits = (0.0, 0.0)
        elif index == (self.function.size() - 1):
            x_limits = (start_value[0], start_value[0])
        else:
            value_at = self.function.value_at
            x_limits = tuple([value_at(i)[0] for i in (index-1, index+1)])

        x_val = min(max(x_limits[0], start_value[0] + delta_x), x_limits[1])
        new_value = (x_val, start_value[1] + delta_y)
        self.function.update(index, new_value)
        self.function_updated = True


class ColorFunctionEditorTool(FunctionEditorTool):
    """ A FuctionEditorTool for a color function.
    """
    ui_adapter_klass = ColorFunctionUIAdapter

    def set_delta(self, value, delta_x, delta_y):
        """ Set the value that is being modified
        """
        index, start_value = value
        if index == 0:
            x_limits = (0.0, 0.0)
        elif index == (self.function.size() - 1):
            x_limits = (start_value[0], start_value[0])
        else:
            value_at = self.function.value_at
            x_limits = tuple([value_at(i)[0] for i in (index-1, index+1)])

        x_pos = min(max(x_limits[0], start_value[0] + delta_x), x_limits[1])
        new_value = (x_pos,) + start_value[1:]
        self.function.update(index, new_value)
        self.function_updated = True
