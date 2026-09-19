import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pyray as rl

from openpilot.selfdrive.ui.layouts.main import MainLayout, MainState
from openpilot.selfdrive.ui.layouts.settings.settings import PanelInfo, PanelType, SettingsLayout
from openpilot.selfdrive.ui.layouts.sidebar import Sidebar
from openpilot.system.ui.lib.application import MouseEvent, MousePos
from openpilot.system.ui.widgets import Widget


class EmptyWidget(Widget):
  def _render(self, rect):
    pass


class TestMainLayout(unittest.TestCase):
  def setUp(self):
    self.gui = SimpleNamespace(mouse_events=[], show_touches=False)
    for target, value in (
      ('openpilot.system.ui.widgets.gui_app', self.gui),
      ('openpilot.system.ui.widgets.device', SimpleNamespace(awake=True)),
      ('openpilot.selfdrive.ui.layouts.main.ui_state', SimpleNamespace(started=False, is_body=False)),
    ):
      self.enterContext(patch(target, value))
    # Exercise real input/navigation handling without textures or a graphics context.
    self.enterContext(patch.object(Sidebar, '_render'))
    self.enterContext(patch.object(Sidebar, '_update_state'))
    self.enterContext(patch.object(SettingsLayout, '_render'))

    sidebar = Sidebar.__new__(Sidebar)
    Widget.__init__(sidebar)
    sidebar._recording_audio = False
    settings = SettingsLayout.__new__(SettingsLayout)
    Widget.__init__(settings)
    settings._current_panel = PanelType.DEVICE
    settings._panels = {PanelType.DEVICE: PanelInfo('Device', EmptyWidget())}
    settings._close_btn_rect = rl.Rectangle(150, 60, 200, 200)

    self.layout = MainLayout.__new__(MainLayout)
    Widget.__init__(self.layout)
    self.layout._sidebar = sidebar
    self.layout._current_mode = MainState.HOME
    self.layout._layouts = {MainState.HOME: EmptyWidget(), MainState.SETTINGS: settings}
    self.layout.set_rect(rl.Rectangle(0, 0, 2160, 1080))
    sidebar.set_callbacks(on_settings=self.layout._on_settings_clicked)
    settings.set_callbacks(on_close=self.layout._set_mode_for_state)

    # This point is inside both the sidebar's Settings and the new Close button.
    pos = MousePos(200, 90)
    self.press = MouseEvent(pos, 0, True, False, True, 1.0)
    self.release = MouseEvent(pos, 0, False, True, False, 1.02)

  def render(self, events):
    self.gui.mouse_events = events
    self.layout._render_main_content()

  def test_short_tap_opens_settings(self):
    self.render([self.press, self.release])
    self.assertEqual(self.layout._current_mode, MainState.SETTINGS)

  def test_tap_across_frames_opens_settings(self):
    self.render([self.press])
    self.render([self.release])
    self.assertEqual(self.layout._current_mode, MainState.SETTINGS)

  def test_next_tap_can_close_settings(self):
    self.layout.open_settings(PanelType.DEVICE)
    self.render([])
    self.render([self.press, self.release])
    self.assertEqual(self.layout._current_mode, MainState.HOME)
