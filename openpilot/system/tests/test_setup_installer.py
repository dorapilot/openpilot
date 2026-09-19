import ast
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import urllib.error
import urllib.request


class TestSetupInstaller(unittest.TestCase):
  def test_installer_handoff(self):
    ui_dir = Path(__file__).resolve().parents[1] / 'ui'
    for board in ('tici', 'mici'):
      tree = ast.parse((ui_dir / f'{board}_setup.py').read_text())
      setup = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Setup')
      download = next(node for node in setup.body if isinstance(node, ast.FunctionDef) and node.name == '_download_thread')
      default_url = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                         and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'OPENPILOT_URL')
      for bundled, custom in ((True, False), (False, False), (True, True)):
        with self.subTest(board=board, bundled=bundled, custom=custom), tempfile.TemporaryDirectory() as tmp:
          root = Path(tmp)
          bundle = root / 'installer_dorapilot'
          remote = root / 'remote'
          if bundled:
            bundle.write_bytes(b'\x7fELFbundled')
          remote.write_bytes(b'\x7fELFremote')
          destination = root / 'installer'
          source_record = root / 'installer_url'
          url = 'https://example.com/custom-installer' if custom else default_url
          app = SimpleNamespace(request_close=Mock())
          state = SimpleNamespace(download_url=url, download_progress=0, download_failed=Mock())
          namespace = {'os': os, 'urllib': urllib, 'time': time, 'HARDWARE': Mock(), 'gui_app': app,
                       'USER_AGENT': 'AGNOSSetup-liberation-day-7.2', 'OPENPILOT_URL': default_url, 'BUNDLED_INSTALLER_PATH': str(bundle),
                       'INSTALLER_DESTINATION_PATH': str(destination), 'INSTALLER_URL_PATH': str(source_record)}
          exec(compile(ast.Module(body=[download], type_ignores=[]), str(ui_dir / f'{board}_setup.py'), 'exec'), namespace)
          urlopen = urllib.request.urlopen

          def open_request(request, urlopen=urlopen, url=url, remote=remote, **kwargs):
            if request.type == 'file':
              return urlopen(request, **kwargs)
            self.assertEqual(request.full_url, url)
            return urlopen(remote.as_uri(), **kwargs)

          with patch.object(urllib.request, 'urlopen', side_effect=open_request) as request:
            namespace['_download_thread'](state)
          if not bundled and not custom:
            request.assert_not_called()
            self.assertFalse(destination.exists())
            self.assertFalse(source_record.exists())
            app.request_close.assert_not_called()
            if board == 'tici':
              state.download_failed.assert_called_once_with(url, 'Bundled dorapilot installer is missing.')
            else:
              self.assertEqual(state._download_failed_reason, 'Bundled dorapilot installer is missing.')
            continue
          self.assertEqual(destination.read_bytes(), b'\x7fELFbundled' if bundled and not custom else b'\x7fELFremote')
          self.assertEqual(source_record.read_text(), url)
          self.assertEqual(state.download_progress, 100)
          state.download_failed.assert_not_called()
          app.request_close.assert_called_once()


if __name__ == '__main__':
  unittest.main()
