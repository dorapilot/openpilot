from types import SimpleNamespace
from unittest.mock import Mock, patch

import cffi

from openpilot.common.test import OpenpilotTestCase
from openpilot.system.ui.lib import egl


class TestEGL(OpenpilotTestCase):
  def test_extension_lookup(self):
    names = (b'eglCreateImageKHR', b'eglDestroyImageKHR', b'glEGLImageTargetTexture2DOES')
    for missing in (None, *names):
      with self.subTest(missing=missing):
        ffi = cffi.FFI()
        library = SimpleNamespace(
          eglGetCurrentDisplay=lambda ffi=ffi: ffi.cast('void *', 1),
          eglGetError=Mock(), glBindTexture=Mock(), glActiveTexture=Mock(),
          eglGetProcAddress=lambda name, missing=missing, ffi=ffi: ffi.cast('void (*)(void)', 0 if name == missing else 1),
        )
        with patch.object(egl, '_egl', egl.EGLState()), patch.object(egl.cffi, 'FFI', return_value=ffi), \
             patch.object(ffi, 'dlopen', return_value=library):
          self.assertEqual(egl.init_egl(), missing is None)
          self.assertEqual(egl._egl.initialized, missing is None)
