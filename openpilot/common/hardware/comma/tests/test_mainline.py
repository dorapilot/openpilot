import tempfile
import errno
import struct
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from openpilot.common import gpio, i2c
from openpilot.common.hardware.base import ThermalZone
from openpilot.common.hardware.comma import amplifier, hardware
from openpilot.common.test import OpenpilotTestCase


class TestMainlineHardware(OpenpilotTestCase):
  def test_power_monitor_discovery(self):
    for mainline in (False, True):
      with self.subTest(mainline=mainline), tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        path = '/sys/bus/platform/devices/a88000.i2c/i2c-10/10-0040/hwmon/hwmon21' if mainline else '/sys/class/hwmon/hwmon1'
        monitor = root / path.lstrip('/')
        monitor.mkdir(parents=True)
        for name, value in [('in1_input', 5040), ('curr1_input', 113), ('power1_input', 568550)]:
          (monitor / name).write_text(str(value))
        def read(path, *args, root=root):
          return open(root / str(path).lstrip('/'), *args)
        with patch.object(Path, 'glob', return_value=iter([Path(path)] if mainline else [])), \
             patch.object(hardware, 'open', side_effect=read, create=True), \
             patch('openpilot.common.hardware.base.open', side_effect=read, create=True):
          device = hardware.HardwareComma()
          self.assertAlmostEqual(device.get_current_power_draw(), .568550)
          self.assertEqual(device.get_voltage(), 5040)
          self.assertEqual(device.get_current(), 113)

  def test_irq_policy(self):
    for mainline in (False, True):
      device = hardware.HardwareComma()
      device.__dict__['amplifier'] = None
      with self.subTest(mainline=mainline), \
           patch.object(hardware.os.path, 'isdir', return_value=mainline), \
           patch.object(hardware.os.path, 'exists', return_value=False), \
           patch.object(hardware, 'affine_irq') as irq, patch.object(hardware, 'sudo_write'), \
           patch.object(hardware, 'gpio_init'), patch.object(hardware, 'gpio_set'), \
           patch.object(hardware.subprocess, 'run'), patch.object(hardware.subprocess, 'call'), \
           patch.object(hardware.subprocess, 'check_output', return_value='123'):
        device.initialize_hardware()
        device.set_power_save(False)
        expected = ([(1, 'venus'), (1, '890000.i2c'), (1, '894000.i2c'), (1, 'a88000.i2c'),
                     (5, 's6sy761_irq'), (3, '880000.spi'), (7, 'gpu-irq')]
                    if mainline else [(1, 'msm_vidc'), (1, 'i2c_geni'), (5, 'fts_ts'), (5, 'msm_drm'), (3, 'spi_geni'), (7, 'kgsl-3d0')])
        expected += [(6, action) for action in ('a5', 'cci', 'cpas_camnoc', 'cpas-cdm', 'csid', 'ife', 'csid-lite', 'ife-lite')]
        self.assertEqual([c.args for c in irq.call_args_list], expected)

  def test_i2c_bus_discovery(self):
    for device, default, bus in [('890000.i2c', 1, 4), ('a88000.i2c', 0, 10), ('890000.i2c', 1, None)]:
      paths = [] if bus is None else [f'/sys/bus/platform/devices/{device}/i2c-{bus}']
      with self.subTest(device=device, bus=bus), patch('glob.glob', return_value=paths):
        self.assertEqual(i2c.get_i2c_bus(device, default), default if bus is None else bus)

  def test_amplifier_mainline_bus(self):
    with patch('glob.glob', return_value=['/sys/bus/platform/devices/a88000.i2c/i2c-10']), \
         patch.object(amplifier.os.path, 'exists', side_effect=lambda p: p == '/dev/i2c-10'):
      amp = amplifier.Amplifier()
      self.assertTrue(amp.available)
      self.assertEqual(amp.AMP_I2C_BUS, 10)

  def test_tlmm_gpiochip_discovery(self):
    for paths, expected in [([], 0), (['/sys/bus/platform/devices/3400000.pinctrl/gpiochip2'], 2)]:
      with self.subTest(paths=paths), patch.object(gpio.globmod, 'glob', return_value=paths):
        self.assertEqual(gpio.get_tlmm_gpiochip(), expected)

  def test_mainline_thermal_config(self):
    with patch.object(hardware.os.path, 'isdir', side_effect=lambda p: p == '/sys/bus/platform/devices/5000000.gpu'), \
         patch.object(hardware.HardwareComma, 'get_device_type', return_value='tizi'):
      config = hardware.HardwareComma().get_thermal_config()
    self.assertEqual([zone.name for zone in config.cpu], [f'cpu{i}-thermal' for i in range(8)])
    self.assertEqual([zone.name for zone in config.gpu], ['gpu-top-thermal', 'gpu-bottom-thermal'])
    self.assertEqual(config.dsp.name, 'q6-hvx-thermal')
    self.assertEqual(config.memory.name, 'mem-thermal')
    self.assertIsNone(config.pmic)
    temperatures = {zone.name: 50. + i for i, zone in enumerate(config.cpu + config.gpu + [config.dsp, config.memory])}
    with patch.object(ThermalZone, 'read', autospec=True, side_effect=lambda zone: temperatures[zone.name]):
      self.assertEqual(config.get_msg(), {'cpuTempC': list(range(50, 58)), 'gpuTempC': [58, 59],
                                         'dspTempC': 60, 'memoryTempC': 61})

  def test_downstream_thermal_config(self):
    with patch.object(hardware.os.path, 'isdir', return_value=False), \
         patch.object(hardware.HardwareComma, 'get_device_type', return_value='mici'):
      config = hardware.HardwareComma().get_thermal_config()
    self.assertEqual([zone.name for zone in config.cpu], [f'cpu{i}-silver-usr' for i in range(4)] +
                                                      [f'cpu{i}-gold-usr' for i in range(4)])
    self.assertEqual([zone.name for zone in config.gpu], ['gpu0-usr', 'gpu1-usr'])
    self.assertEqual(config.intake.name, 'intake')
    self.assertEqual(config.exhaust.name, 'exhaust')

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
    with patch.object(gpio.os, 'uname', return_value=SimpleNamespace(release='4.9.103')), \
         patch.object(gpio.os, 'open', return_value=16), patch.object(gpio.os, 'close') as close, \
         patch.object(gpio.fcntl, 'ioctl', side_effect=ioctl):
      self.assertEqual(gpio.gpiochip_get_ro_value_fd('test', 0, 8), 17)
      close.assert_called_once_with(16)

  def test_gpiochip_v2_request(self):
    # The kernel UAPI defines a 592-byte request with config at 288 and num_lines at 560.
    def ioctl(_fd, request, data):
      if request != 0xc250b407:
        raise OSError(errno.EINVAL, 'GPIO v1 is disabled')
      encoded = bytes(data)
      self.assertEqual(len(encoded), 592)
      self.assertEqual(struct.unpack_from('<I', encoded, 0)[0], 84)
      self.assertEqual(encoded[256:288].rstrip(b'\0'), b'sensord')
      self.assertEqual(struct.unpack_from('<Q', encoded, 288)[0], 0x34)
      self.assertEqual(encoded[296:560], bytes(264))
      self.assertEqual(struct.unpack_from('<I', encoded, 560)[0], 1)
      self.assertEqual(encoded[564:588], bytes(24))
      data.fd = 17
    for release in ('5.10.0', '7.2.0-vamos-4ec4055'):
      with self.subTest(release=release), patch.object(gpio.os, 'uname', return_value=SimpleNamespace(release=release)), \
           patch.object(gpio.os, 'open', return_value=16), patch.object(gpio.os, 'close') as close, \
           patch.object(gpio.fcntl, 'ioctl', side_effect=ioctl):
        self.assertEqual(gpio.gpiochip_get_ro_value_fd('sensord', 2, 84), 17)
        close.assert_called_once_with(16)

  def test_gpiochip_request_error(self):
    for release in ('4.9.103', '7.2.0-vamos-4ec4055'):
      with self.subTest(release=release), patch.object(gpio.os, 'uname', return_value=SimpleNamespace(release=release)), \
           patch.object(gpio.os, 'open', return_value=16), patch.object(gpio.os, 'close') as close, \
           patch.object(gpio.fcntl, 'ioctl', side_effect=OSError(errno.EACCES, 'denied')):
        with self.assertRaises(OSError) as error:
          gpio.gpiochip_get_ro_value_fd('sensord', 2, 84)
        self.assertEqual(error.exception.errno, errno.EACCES)
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
