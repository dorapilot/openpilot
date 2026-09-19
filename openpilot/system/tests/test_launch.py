import os
from pathlib import Path
import subprocess
import tempfile
import unittest


BASEDIR = Path(__file__).resolve().parents[3]


class TestLaunch(unittest.TestCase):
  def test_os_version_gate(self):
    for os_id in ('ubuntu', 'void'):
      for version, verified in (('liberation-day-7.2', False), ('19.7', False), ('19.7', True)):
        with self.subTest(os_id=os_id, version=version, verified=verified), tempfile.TemporaryDirectory() as tmp:
          root = Path(tmp).resolve()
          tmp = str(root)
          (root / 'AGNOS').touch()
          (root / 'tmp').mkdir()
          (root / 'VERSION').write_text(version)
          (root / 'os-release').write_text(f'ID={os_id}\n')
          (root / 'launch_env.sh').write_text((BASEDIR / 'launch_env.sh').read_text())
          script = (BASEDIR / 'launch_chffrplus.sh').read_text()
          for path in ('/VERSION', '/AGNOS', '/dev/adsprpc-smd', '/dev/ion', '/dev/kgsl-3d0', '/data/', '/tmp/launch_log'):
            script = script.replace(path, tmp + path)
          script = script.replace('/etc/os-release', str(root / 'os-release'))
          (root / 'launch_chffrplus.sh').write_text(script)
          agnos = root / 'openpilot/common/hardware/comma'
          agnos.mkdir(parents=True)
          env = dict(os.environ)
          env.pop('AGNOS_VERSION', None)
          stubs = f'''
function {agnos}/agnos.py {{ echo verify; return {0 if verified else 1}; }}
function {agnos}/updater {{ echo updater; exit 0; }}
'''
          result = subprocess.run(['bash', '-c', stubs + '''
rm() { :; }
ln() { :; }
sudo() { echo "$*"; if [ "$1" = reboot ]; then exit 0; fi; }
tmux() { echo manager >&2; exit 0; }
source ./launch_chffrplus.sh
'''], cwd=root, env=env, capture_output=True, text=True, timeout=10, check=True)
          self.assertIn('abctl --set_success', result.stdout)
          if version == 'liberation-day-7.2':
            self.assertNotIn('verify', result.stdout)
            self.assertIn('manager', result.stderr)
          else:
            self.assertIn('verify', result.stdout)
            self.assertIn('reboot' if verified else 'updater', result.stdout)
            self.assertNotIn('manager', result.stderr)


if __name__ == '__main__':
  unittest.main()
