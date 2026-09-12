import tempfile
from pathlib import Path
from unittest.mock import patch

from openpilot.common import gpio
from openpilot.common.hardware.comma import amplifier, hardware
from openpilot.common.test import OpenpilotTestCase


class TestMainlineHardware(OpenpilotTestCase):
  def test_sysfs_gpio_numbering(self):
    for base in (0, 512):
      with self.subTest(base=base), tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        pin = root / f'sys/class/gpio/gpio{base + 8}'
        pin.mkdir(parents=True)
        base_path = '/sys/bus/platform/devices/3400000.pinctrl/gpio/gpiochip0/base'
        if base:
          (root / base_path.lstrip('/')).parent.mkdir(parents=True)
          (root / base_path.lstrip('/')).write_text(str(base))
        gpio.get_tlmm_base.cache_clear()
        with patch.object(gpio.globmod, 'glob', return_value=[base_path] if base else []), \
             patch.object(gpio, 'open', side_effect=lambda path, *args, root=root: open(root / path.lstrip('/'), *args), create=True):
          gpio.gpio_export(8)
          gpio.gpio_init(8, True)
          gpio.gpio_set(8, True)
          self.assertTrue(gpio.gpio_read(8))
          self.assertEqual((pin.parent / 'export').read_text(), str(base + 8))
          self.assertEqual((pin / 'direction').read_bytes(), b'out')
          self.assertEqual((pin / 'value').read_bytes(), b'1')
        gpio.get_tlmm_base.cache_clear()

  def test_gpiochip_keeps_line_offset(self):
    def ioctl(_fd, _request, data):
      self.assertEqual(data.lineoffset, 8)
      data.fd = 17
    with patch.object(gpio.os, 'open', return_value=16), patch.object(gpio.os, 'close') as close, \
         patch.object(gpio.fcntl, 'ioctl', side_effect=ioctl):
      self.assertEqual(gpio.gpiochip_get_ro_value_fd('test', 0, 8), 17)
      close.assert_called_once_with(16)

  def test_backlight_paths(self):
    for name in ('panel0-backlight', 'ae94000.dsi.0'):
      with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        backlight = root / f'sys/class/backlight/{name}'
        backlight.mkdir(parents=True)
        (backlight / 'max_brightness').write_text('200')
        with patch.object(hardware.os.path, 'isdir', side_effect=lambda path, root=root: (root / path.lstrip('/')).is_dir()), \
             patch.object(hardware, 'open', side_effect=lambda path, *args, root=root: open(root / path.lstrip('/'), *args), create=True):
          device = hardware.HardwareComma()
          device.set_screen_brightness(50)
          self.assertEqual((backlight / 'brightness').read_text(), '100')
          self.assertEqual(device.get_screen_brightness(), 50)
          device.set_display_power(False)
          self.assertEqual((backlight / 'bl_power').read_text(), '4')
          device.set_display_power(True)
          self.assertEqual((backlight / 'bl_power').read_text(), '0')

  def test_missing_amplifier_bus(self):
    with patch.object(amplifier.os.path, 'exists', return_value=False), patch.object(amplifier, 'SMBus') as bus:
      self.assertFalse(amplifier.Amplifier().set_global_shutdown(True))
      bus.assert_not_called()
