import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from openpilot.common.spinner import Spinner


class TestSpinner(unittest.TestCase):
  def test_close_allows_child_cleanup(self):
    with tempfile.TemporaryDirectory() as tmp:
      cleaned = Path(tmp) / 'cleaned'
      script = """
import atexit
from pathlib import Path
import sys

atexit.register(Path(sys.argv[1]).write_text, 'cleaned')
print('ready', flush=True)
sys.stdin.read()
"""
      with subprocess.Popen([sys.executable, '-c', script, str(cleaned)], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as child:
        try:
          self.assertEqual(child.stdout.readline().strip(), 'ready')
          with patch('openpilot.common.spinner.subprocess.Popen', return_value=child):
            spinner = Spinner()
          spinner.close()
          self.assertEqual(child.returncode, 0)
          self.assertEqual(cleaned.read_text(), 'cleaned')
          spinner.close()
        finally:
          if child.poll() is None:
            child.kill()
            child.wait(timeout=5)

  def test_close_kills_unresponsive_child(self):
    script = """
import signal

signal.signal(signal.SIGINT, signal.SIG_IGN)
print('ready', flush=True)
signal.pause()
"""
    with subprocess.Popen([sys.executable, '-c', script], stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as child:
      try:
        self.assertEqual(child.stdout.readline().strip(), 'ready')
        with patch('openpilot.common.spinner.subprocess.Popen', return_value=child):
          spinner = Spinner()
        started = time.monotonic()
        spinner.close()
        self.assertEqual(child.poll(), -signal.SIGKILL)
        self.assertLess(time.monotonic() - started, 20)
        spinner.close()
      finally:
        if child.poll() is None:
          child.kill()
          child.wait(timeout=5)
