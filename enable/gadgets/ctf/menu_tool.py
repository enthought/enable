import json

from enable.component import Component
from enable.gadgets.ctf.piecewise import PiecewiseFunction, verify_values
from enable.gadgets.ctf.utils import (
    FunctionUIAdapter, AlphaFunctionUIAdapter, ColorFunctionUIAdapter
)
from enable.tools.pyface.context_menu_tool import ContextMenuTool
from pyface.action.api import Action, Group, MenuManager, Separator
from traits.api import Callable, Instance, List, Type


class BaseCtfAction(Action):
    component = Instance(Component)
    function = Instance(PiecewiseFunction)
    ui_adaptor = Instance(FunctionUIAdapter)
    ui_adaptor_klass = Type

    def _ui_adaptor_default(self):
        return self.ui_adaptor_klass(component=self.component,
                                     function=self.function)

    def _get_relative_event_position(self, event):
        return self.ui_adaptor.screen_to_function((event.x, event.y))


class AddColorAction(BaseCtfAction):
    name = 'Add Color...'
    ui_adaptor_klass = ColorFunctionUIAdapter

    # A callable which prompts the user for a color
    prompt_color = Callable

    def perform(self, event):
        pos = self._get_relative_event_position(event.enable_event)
        color_val = (pos[0],) + self.prompt_color()
        self.component.add_function_node(self.function, color_val)


class AddOpacityAction(BaseCtfAction):
    name = 'Add Opacity'
    ui_adaptor_klass = AlphaFunctionUIAdapter

    def perform(self, event):
        pos = self._get_relative_event_position(event.enable_event)
        self.component.add_function_node(self.function, pos)


class EditColorAction(BaseCtfAction):
    name = 'Edit Color...'
    ui_adaptor_klass = ColorFunctionUIAdapter

    # A callable which prompts the user for a color
    prompt_color = Callable

    def perform(self, event):
        mouse_pos = (event.enable_event.x, event.enable_event.y)
        index = self.ui_adaptor.function_index_at_position(*mouse_pos)
        if index is not None:
            color_val = self.function.value_at(index)
            new_value = (color_val[0],) + self.prompt_color(color_val[1:])
            self.component.edit_function_node(self.function, index, new_value)


class RemoveNodeAction(Action):
    """ Removes a node from one of the functions.
    """
    name = 'Remove Node'
    component = Instance(Component)
    alpha_func = Instance(PiecewiseFunction)
    color_func = Instance(PiecewiseFunction)
    ui_adaptors = List(Instance(FunctionUIAdapter))

    def _ui_adaptors_default(self):
        # Alpha function first so that it will take precedence in removals.
        return [AlphaFunctionUIAdapter(component=self.component,
                                       function=self.alpha_func),
                ColorFunctionUIAdapter(component=self.component,
                                       function=self.color_func)]

    def perform(self, event):
        mouse_pos = (event.enable_event.x, event.enable_event.y)
        for adaptor in self.ui_adaptors:
            index = adaptor.function_index_at_position(*mouse_pos)
            if index is not None:
                self.component.remove_function_node(adaptor.function, index)
                return


class LoadFunctionAction(Action):
    name = 'Load Function...'
    component = Instance(Component)
    alpha_func = Instance(PiecewiseFunction)
    color_func = Instance(PiecewiseFunction)

    # A callable which prompts the user for a filename
    prompt_filename = Callable

    def perform(self, event):
        filename = self.prompt_filename(action='open')
        with open(filename, 'r') as fp:
            loaded_data = json.load(fp)

        # Sanity check
        if not self._verify_loaded_data(loaded_data):
            return

        parts = (('alpha', self.alpha_func), ('color', self.color_func))
        for name, func in parts:
            func.clear()
            for value in loaded_data[name]:
                func.insert(tuple(value))
        self.component.update_function()

    def _verify_loaded_data(self, data):
        keys = ('alpha', 'color')
        has_values = all(k in data for k in keys)
        return has_values and all(verify_values(data[k]) for k in keys)


class SaveFunctionAction(Action):
    name = 'Save Function...'
    component = Instance(Component)
    alpha_func = Instance(PiecewiseFunction)
    color_func = Instance(PiecewiseFunction)

    # A callable which prompts the user for a filename
    prompt_filename = Callable

    def perform(self, event):
        filename = self.prompt_filename(action='save')
        function = {'alpha': self.alpha_func.values(),
                    'color': self.color_func.values()}
        with open(filename, 'w') as fp:
            json.dump(function, fp, indent=1)


class FunctionMenuTool(ContextMenuTool):
    def _menu_manager_default(self):
        component = self.component
        alpha_func = component.opacities
        color_func = component.colors
        prompt_color = component.prompt_color_selection
        prompt_filename = component.prompt_file_selection
        return MenuManager(
            Group(
                AddColorAction(component=component, function=color_func,
                               prompt_color=prompt_color),
                AddOpacityAction(component=component, function=alpha_func),
                id='AddGroup',
            ),
            Separator(),
            Group(
                EditColorAction(component=component, function=color_func,
                                prompt_color=prompt_color),
                id='EditGroup',
            ),
            Separator(),
            Group(
                RemoveNodeAction(component=component, alpha_func=alpha_func,
                                 color_func=color_func),
                id='RemoveGroup',
            ),
            Separator(),
            Group(
                LoadFunctionAction(component=component, alpha_func=alpha_func,
                                   color_func=color_func,
                                   prompt_filename=prompt_filename),
                SaveFunctionAction(component=component, alpha_func=alpha_func,
                                   color_func=color_func,
                                   prompt_filename=prompt_filename),
                id='IOGroup',
            ),
        )
