import tempfile
import uuid
from unittest.mock import MagicMock, patch

import numpy as np
from msgq.visionipc import VisionBuf, VisionIpcClient, VisionIpcServer
from tinygrad import Tensor
from tinygrad.engine.jit import TinyJit

from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.modeld.compile_dm_warp import make_warp_dm
from openpilot.selfdrive.modeld.compile_modeld import NV12Frame, MODELD_INPUTS, make_input_queues, make_run_model, make_warp, nv12_copy_size
from openpilot.selfdrive.modeld.dmonitoringmodeld import ModelState
from openpilot.selfdrive.modeld.helpers import dump_oob, load_oob
from openpilot.selfdrive.modeld.modeld import ModelState as DrivingModelState
from openpilot.selfdrive.modeld.parse_model_outputs import Parser
from openpilot.system.camerad.cameras.nv12_info import get_nv12_info


class TestVisionBufInput(OpenpilotTestCase):
  def test_driving_model_import_cache_and_cpu_fallback(self):
    def warp(tfm, big_tfm, frame, big_frame):
      return frame[:4].cat(big_frame[:4]).cast('float32').reshape(1, 8)

    frame_size = 64
    shapes = {'img': (1, 12, 2, 2), 'features_buffer': (1, 2, 4), 'desire_pulse': (1, 3, 8),
              'traffic_convention': (1, 2), 'action_t': (1, 2)}
    for direct in (False, True):
      with self.subTest(direct=direct):
        model = object.__new__(DrivingModelState)
        model.model_device = 'CPU'
        model.direct_camera_frames = direct
        model._camera_server_id = None
        model._blob_cache = {}
        model.frame_copy_size = frame_size
        model.input_queues, model.npy, model.frame_views = make_input_queues(shapes, 2, 'CPU', frame_size, direct_camera_frames=direct)
        model.prev_desire = np.zeros(8, dtype=np.float32)
        model.chestnut = False
        model.output_slices = {'hidden_state': slice(0, 4), 'pixels': slice(4, 8)}
        model.parser = Parser(ignore_missing=True)
        model.run_model = TinyJit(make_run_model(warp, lambda warped, *args: (warped,), {'input_shapes': shapes}, frame_size), prune=True)

        # This unit test supplies DMA-BUF metadata at the VisionBuf boundary; CPU imports use real shared-memory FDs.
        name = f'test_driving_import_{uuid.uuid4().hex}'
        server = VisionIpcServer(name)
        server.create_buffers_with_sizes(0, 1, 8, 4, frame_size, 8, 32)
        server.start_listener()
        client = VisionIpcClient(name, 0, False)
        self.assertTrue(client.connect(True))
        with patch.object(Tensor, 'from_blob', wraps=Tensor.from_blob) as from_blob:
          for value in (16, 64, 128, 235):
            frame = np.full(frame_size, value, dtype=np.uint8)
            server.send(0, frame)
            received = client.recv()
            self.assertIsNotNone(received)
            buf = MagicMock(spec=VisionBuf, data=received.data, is_dma_buf=True, server_id=received.server_id, idx=received.idx, fd=received.fd)
            buf.cpu_access.return_value.__enter__.return_value = buf.data
            inputs = {'desire_pulse': np.zeros(8), 'traffic_convention': np.zeros(2), 'action_t': np.zeros(2)}
            with received.cpu_access():
              outputs = model.run({'img': buf, 'big_img': buf}, dict.fromkeys(('img', 'big_img'), np.eye(3)), inputs)
            np.testing.assert_array_equal(outputs['pixels'], np.full((1, 4), value))
            if direct:
              buf.cpu_access.assert_not_called()
            else:
              self.assertEqual(buf.cpu_access.call_count, 2)
          self.assertEqual(from_blob.call_count, 2 if direct else 0)
          if direct:
            buf.server_id += 1
            with received.cpu_access():
              model.run({'img': buf, 'big_img': buf}, dict.fromkeys(('img', 'big_img'), np.eye(3)), inputs)
            self.assertEqual(from_blob.call_count, 4)
            buf.is_dma_buf = False
            with self.assertRaisesRegex(RuntimeError, 'requires a DMA-BUF'):
              model.run({'img': buf}, {}, {})
          dummy = np.full(frame_size, 42, dtype=np.uint8)
          outputs = model.run({'img': dummy, 'big_img': dummy}, dict.fromkeys(('img', 'big_img'), np.eye(3)), inputs)
          np.testing.assert_array_equal(outputs['pixels'], np.full((1, 4), 42))

  def test_driving_warp_uses_imported_frames_after_jit_reload(self):
    width, height, model_w, model_h = 64, 32, 16, 8
    layout = get_nv12_info(width, height)
    stride, y_height, uv_height, size = layout
    shapes = {'img': (1, 12, model_h // 2, model_w // 2), 'features_buffer': (1, 2, 4), 'desire_pulse': (1, 3, 8),
              'traffic_convention': (1, 2), 'action_t': (1, 2)}
    frame_size = nv12_copy_size(stride, y_height, uv_height)
    queues, npy, _ = make_input_queues(shapes, 2, 'CPU', frame_size, direct_camera_frames=True)
    npy['tfm'][:] = npy['big_tfm'][:] = np.eye(3, dtype=np.float32)
    self.assertLess(queues['packed_npy_inputs'].numel(), frame_size)
    warp = make_warp(NV12Frame(width, height, *layout), model_w, model_h)
    run = TinyJit(make_run_model(warp, lambda warped, *args: (warped,), {'input_shapes': shapes}, frame_size), prune=True)
    name = f'test_driving_{uuid.uuid4().hex}'
    server = VisionIpcServer(name)
    server.create_buffers_with_sizes(0, 1, width, height, size, stride, stride * y_height)
    server.start_listener()
    client = VisionIpcClient(name, 0, False)
    self.assertTrue(client.connect(True))
    imported = None
    imported_extra = None
    for index, value in enumerate((16, 64, 128, 235, 32)):
      frame = np.full(size, 128, dtype=np.uint8)
      frame[:stride * height].reshape(height, stride)[:, :width] = value
      server.send(0, frame)
      buf = client.recv()
      self.assertIsNotNone(buf)
      if imported is None:
        imported = Tensor.from_blob(np.frombuffer(buf.data, dtype=np.uint8).ctypes.data, (frame_size,), fd=buf.fd, dtype='uint8', device='CPU')
        imported_extra = Tensor.from_blob(np.frombuffer(buf.data, dtype=np.uint8).ctypes.data, (frame_size,), fd=buf.fd, dtype='uint8', device='CPU')
      with buf.cpu_access():
        out, = run(**{k: queues[k] for k in MODELD_INPUTS}, frame=imported, big_frame=imported_extra)
        actual = out.numpy()
      expected = np.full((2, 6, model_h // 2, model_w // 2), 128, dtype=np.uint8)
      expected[:, :4] = value
      np.testing.assert_array_equal(actual, expected)
      if index == 2:
        with tempfile.TemporaryFile() as f:
          dump_oob(run, f)
          f.seek(0)
          run = load_oob(f)

  def test_driver_monitoring_import_and_ring_reuse(self):
    width, height = 64, 32
    layout = get_nv12_info(width, height)
    stride, y_height, _, size = layout
    name = f'test_dm_{uuid.uuid4().hex}'
    server = VisionIpcServer(name)
    server.create_buffers_with_sizes(0, 1, width, height, size, stride, stride * y_height)
    server.start_listener()
    client = VisionIpcClient(name, 0, False)
    self.assertTrue(client.connect(True))

    # Exercise the real import and warp without needing trained model artifacts.
    model = object.__new__(ModelState)
    model.DEV = 'CPU'
    model.frame_buf_params = layout
    model.numpy_inputs = {'calib': np.zeros((1, 3), dtype=np.float32)}
    model.warp_inputs_np = {'transform': np.zeros((3, 3), dtype=np.float32)}
    model.warp_inputs = {k: Tensor(v, device='NPY') for k, v in model.warp_inputs_np.items()}
    model.tensor_inputs = {}
    model._blob_cache = {}
    model.image_warp = TinyJit(make_warp_dm(NV12Frame(width, height, *layout), width, height))
    model.model_run = lambda **inputs: inputs['input_img']

    with patch.object(Tensor, 'from_blob', wraps=Tensor.from_blob) as from_blob:
      for value in (16, 64, 128, 235):
        frame = np.full(size, 128, dtype=np.uint8)
        frame[:stride * height].reshape(height, stride)[:, :width] = value
        server.send(0, frame)
        buf = client.recv()
        self.assertIsNotNone(buf)
        output, _ = model.run(buf, np.array([1, 2, 3]), np.eye(3, dtype=np.float32))
        np.testing.assert_array_equal(output, np.full(width * height, value))
        np.testing.assert_array_equal(model.numpy_inputs['calib'], [[1, 2, 3]])
        self.assertEqual(from_blob.call_args.kwargs['fd'], buf.fd)
      self.assertEqual(from_blob.call_count, 1)
    model._blob_cache.clear()
