import struct
from types import SimpleNamespace
from unittest.mock import Mock, patch

from openpilot.common.test import OpenpilotTestCase
from openpilot.system.sensord import sensord


class TestSensorPlatform(OpenpilotTestCase):
  def test_gpio_timestamp_clock(self):
    wall, mono = 1_700_000_000_000_000_000, 10_000_000_000
    for release in ('4.9.103', '5.6.0', '5.7.0', '5.10.0', '7.2.0-vamos-4ec4055'):
      with self.subTest(release=release):
        old_clock = release.startswith(('4.', '5.6.'))
        event_size = 48 if release.startswith(('5.10.', '7.')) else 16
        data = struct.pack('<QI', (wall if old_clock else mono) - 1_000_000, 1) + bytes(event_size - 12)
        sensor, stop, poller = Mock(), Mock(), Mock()
        stop.is_set.side_effect = [False, True]
        poller.poll.return_value = [(17, sensord.select.POLLIN)]
        with patch.object(sensord.os, 'uname', return_value=SimpleNamespace(release=release)), \
             patch.object(sensord, 'gpiochip_get_ro_value_fd', return_value=17), \
             patch.object(sensord, 'get_tlmm_gpiochip', return_value=2, create=True), \
             patch.object(sensord, 'get_irqs_for_action', return_value=['420'], create=True), \
             patch.object(sensord, 'sudo_write') as write, \
             patch.object(sensord.os.path, 'exists', return_value=False), \
             patch.object(sensord.os, 'read', return_value=bytes(data)), \
             patch.object(sensord.os, 'close') as close, \
             patch.object(sensord.select, 'poll', return_value=poller), \
             patch.object(sensord.time, 'time_ns', return_value=wall), \
             patch.object(sensord.time, 'monotonic_ns', return_value=mono), \
             patch.object(sensord.messaging, 'PubMaster'), patch.object(sensord.messaging, 'new_message'):
          sensord.interrupt_loop([(sensor, 'accelerometer', True)], stop)
        sensor.get_event.assert_called_once_with(mono - 1_000_000)
        write.assert_called_once_with('1\n', '/proc/irq/420/smp_affinity_list')
        close.assert_called_once_with(17)
