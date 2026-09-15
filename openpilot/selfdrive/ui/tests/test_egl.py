from types import SimpleNamespace
from unittest.mock import Mock, patch

import cffi

from openpilot.common.test import OpenpilotTestCase
from openpilot.system.ui.lib import egl


class TestEGL(OpenpilotTestCase):
  def test_external_texture_ownership(self):
    ffi = cffi.FFI()
    textures = {42: '2D'}
    bound = [0]
    sampled = []

    def generate(count, names):
      self.assertEqual(count, 1)
      names[0] = max(textures) + 1
      textures[names[0]] = None

    def bind(target, name):
      self.assertEqual(target, egl.GL_TEXTURE_EXTERNAL_OES)
      self.assertIn(textures[name], (None, target), 'texture name already belongs to another target')
      textures[name] = target
      bound[0] = name

    def sample(target, image):
      self.assertEqual(target, egl.GL_TEXTURE_EXTERNAL_OES)
      sampled.append((bound[0], image))

    def delete(count, names):
      self.assertEqual(count, 1)
      del textures[names[0]]

    state = egl.EGLState(initialized=True, ffi=ffi, gles_lib=SimpleNamespace(glGenTextures=generate, glDeleteTextures=delete),
                         active_texture=Mock(), bind_texture=bind, image_target_texture=sample, destroy_image_khr=Mock())
    images = [egl.EGLImage(ffi.cast('void *', n), n + 30) for n in (1, 2)]
    with patch.object(egl, '_egl', state), patch.object(egl.os, 'close') as close:
      for index in (0, 1, 0):
        egl.bind_egl_image(images[index])
      self.assertEqual(sampled[0], sampled[2])
      self.assertNotEqual(sampled[0][0], sampled[1][0])
      self.assertEqual([item[1] for item in sampled], [images[i].egl_image for i in (0, 1, 0)])
      for image in images:
        egl.destroy_egl_image(image)
      self.assertEqual(textures, {42: '2D'})
      self.assertEqual(close.call_count, 2)

  def test_extension_lookup(self):
    names = (b'eglCreateImageKHR', b'eglDestroyImageKHR', b'glEGLImageTargetTexture2DOES')
    for missing in (None, *names):
      with self.subTest(missing=missing):
        ffi = cffi.FFI()
        library = SimpleNamespace(
          eglGetCurrentDisplay=lambda ffi=ffi: ffi.cast('void *', 1),
          eglQueryString=lambda *args, ffi=ffi: ffi.new('char[]', b'EGL_EXT_image_dma_buf_import'),
          eglGetError=Mock(), glBindTexture=Mock(), glActiveTexture=Mock(),
          eglGetProcAddress=lambda name, missing=missing, ffi=ffi: ffi.cast('void (*)(void)', 0 if name == missing else 1),
        )
        with patch.object(egl, '_egl', egl.EGLState()), patch.object(egl.cffi, 'FFI', return_value=ffi), \
             patch.object(ffi, 'dlopen', return_value=library):
          self.assertEqual(egl.init_egl(), missing is None)
          self.assertEqual(egl._egl.initialized, missing is None)

  def test_linear_dma_buf_import(self):
    ffi = cffi.FFI()
    for supported in (False, True):
      with self.subTest(modifiers_supported=supported):
        def create_image(display, context, target, buffer, attrs, supported=supported):
          values = {}
          for index in range(0, 40, 2):
            if attrs[index] == egl.EGL_NONE:
              break
            values[attrs[index]] = attrs[index + 1]
          modifiers = {key: values[key] for key in (0x3443, 0x3444, 0x3445, 0x3446) if key in values}
          self.assertEqual(modifiers, dict.fromkeys((0x3443, 0x3444, 0x3445, 0x3446), 0) if supported else {})
          return ffi.cast('void *', 1)
        state = egl.EGLState(initialized=True, ffi=ffi, display=ffi.NULL, NO_CONTEXT=ffi.NULL,
                             NO_IMAGE_KHR=ffi.NULL, create_image_khr=create_image)
        state.supports_dma_buf_modifiers = supported
        with patch.object(egl, '_egl', state), patch.object(egl.os, 'dup', return_value=42), patch.object(egl.os, 'close') as close:
          image = egl.create_egl_image(1928, 1208, 2048, 7, 2490368)
          self.assertIsNotNone(image)
          self.assertEqual(image.fd, 42)
          close.assert_not_called()
