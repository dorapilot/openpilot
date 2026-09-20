#!/usr/bin/env python3
from pathlib import Path
import unittest
import zipfile


class TestPack(unittest.TestCase):
  def test_updater_contains_current_mici_setup(self):
    root = Path(__file__).resolve().parents[2]
    setup_path = root / 'openpilot/system/ui/mici_setup.py'
    with zipfile.ZipFile(root / 'openpilot/common/hardware/comma/updater') as updater:
      self.assertEqual(updater.read('openpilot/system/ui/mici_setup.py'), setup_path.read_bytes())


if __name__ == '__main__':
  unittest.main()
