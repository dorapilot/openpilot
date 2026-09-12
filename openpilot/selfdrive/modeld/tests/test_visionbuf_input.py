import uuid
from unittest.mock import patch

import numpy as np
from msgq.visionipc import VisionIpcClient, VisionIpcServer
from tinygrad import Tensor
from tinygrad.engine.jit import TinyJit

from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.modeld.compile_dm_warp import make_warp_dm
from openpilot.selfdrive.modeld.compile_modeld import NV12Frame
from openpilot.selfdrive.modeld.dmonitoringmodeld import ModelState
from openpilot.system.camerad.cameras.nv12_info import get_nv12_info


class TestVisionBufInput(OpenpilotTestCase):
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
