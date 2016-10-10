import mock
import nose

from traitsui.tests._tools import skip_if_null

from enable.component import Component
from enable.testing import EnableTestAssistant, _MockWindow


def test_mouse_move():
    test_assistant = EnableTestAssistant()
    component = Component(bounds=[100, 200])
    event = test_assistant.mouse_move(component, 10, 20)

    nose.tools.assert_equal(event.x, 10)
    nose.tools.assert_equal(event.y, 20)
    assert isinstance(event.window, _MockWindow)
    assert not event.alt_down
    assert not event.control_down
    assert not event.shift_down
    nose.tools.assert_equal(event.window.get_pointer_position(), (10, 20))

@skip_if_null
def test_mouse_move_real_window():
    from enable.api import Window

    test_assistant = EnableTestAssistant()
    component = Component(bounds=[100, 200])
    window = Window(None, component=component)

    event = test_assistant.mouse_move(component, 10, 20, window)

    nose.tools.assert_equal(event.x, 10)
    nose.tools.assert_equal(event.y, 20)
    nose.tools.assert_equal(event.window, window)
    assert not event.alt_down
    assert not event.control_down
    assert not event.shift_down
    # can't test pointer position, not set, but if we get here it didn't
    # try to set the pointer position

@skip_if_null
def test_mouse_move_real_window_mocked_position():
    from enable.api import Window

    test_assistant = EnableTestAssistant()
    component = Component(bounds=[100, 200])

    with mock.patch.object(Window, 'get_pointer_position',
                            return_value=None):
        window = Window(None, component=component)
        event = test_assistant.mouse_move(component, 10, 20, window)

        nose.tools.assert_equal(event.x, 10)
        nose.tools.assert_equal(event.y, 20)
        nose.tools.assert_equal(event.window, window)
        assert not event.alt_down
        assert not event.control_down
        assert not event.shift_down
        nose.tools.assert_equal(event.window.get_pointer_position(), (10, 20))
