import queue
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from openpilot.cereal import messaging
from openpilot.common.hardware import COMMA_HARDWARE
from openpilot.common.parameterized import parameterized
from openpilot.common.params import Params
from openpilot.common.test import OpenpilotTestCase
from openpilot.system.hardware.hardwared import hardware_thread


@unittest.skipIf(COMMA_HARDWARE, "hardware_thread integration test uses PC hardware")
class TestHardwared(OpenpilotTestCase):
  @parameterized.expand([False, True])
  def test_engagement_log_transitions(self, fail_first_write):
    pm = messaging.PubMaster(['pandaStates', 'selfdriveState'])
    sm = messaging.SubMaster(['deviceState'])
    params = Params()
    end_event = threading.Event()

    with tempfile.TemporaryDirectory() as tmp:
      log_path = Path(tmp) / 'kmsg'

      def open_kmsg(path, *args, **kwargs):
        nonlocal fail_first_write
        if path == '/dev/kmsg':
          if fail_first_write:
            fail_first_write = False
            raise PermissionError(path)
          return log_path.open('a')
        return open(path, *args, **kwargs)

      with patch('openpilot.system.hardware.hardwared.open', open_kmsg, create=True), ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(hardware_thread, end_event, queue.Queue())
        try:
          for enabled in [False, True, False]:
            updates = 0
            deadline = time.monotonic() + 10
            while updates < 3:
              if future.done():
                future.result()
              self.assertLess(time.monotonic(), deadline, 'hardwared stopped publishing deviceState')
              msg = messaging.new_message('selfdriveState')
              msg.selfdriveState.enabled = enabled
              pm.send('selfdriveState', msg)
              pm.send('pandaStates', messaging.new_message('pandaStates', 0))
              sm.update(100)
              if sm.updated['deviceState']:
                updates += 1
                self.assertEqual(params.get_bool('IsEngaged'), enabled)
        finally:
          end_event.set()
          future.result(timeout=5)

      self.assertEqual(log_path.read_text().splitlines(), [
        '<3>[hardware] engaged: False',
        '<3>[hardware] engaged: True',
        '<3>[hardware] engaged: False',
      ])
