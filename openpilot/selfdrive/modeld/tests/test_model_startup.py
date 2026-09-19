import pickle
import tempfile
import uuid
from pathlib import Path
from unittest.mock import patch

import numpy as np
from msgq.visionipc import VisionIpcClient, VisionIpcServer
from tinygrad import Tensor
from tinygrad.engine.jit import TinyJit, link_linear

from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.modeld.compile_modeld import CAMERA_INPUTS, MODELD_INPUTS, make_input_queues, make_run_model, nv12_copy_size
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.modeld.dmonitoringmodeld import ModelState as DriverMonitoringModelState
from openpilot.selfdrive.modeld.helpers import dump_oob
from openpilot.selfdrive.modeld.modeld import ModelState
from openpilot.selfdrive.modeld.parse_model_outputs import Parser
from openpilot.system.camerad.cameras.nv12_info import get_nv12_info


class TestModelStartup(OpenpilotTestCase):
  def test_driver_monitoring_is_prepared_before_camera_input(self):
    width, height = 64, 32
    stride, y_height, _, size = get_nv12_info(width, height)
    warp = TinyJit(lambda frame, transform: (frame[:4].cast('float32') + transform.to('CPU').sum()).realize())
    run = TinyJit(lambda calib, input_img: (input_img + calib.to('CPU').sum()).realize())
    frame = Tensor(np.zeros(size, dtype=np.uint8), device='CPU').realize()
    transform = Tensor(np.zeros((3, 3), dtype=np.float32), device='NPY').realize()
    calib = Tensor(np.zeros((1, 3), dtype=np.float32), device='NPY').realize()
    for _ in range(3):
      run(calib=calib, input_img=warp(frame, transform)).numpy()

    with tempfile.TemporaryDirectory() as tmp:
      directory = Path(tmp)
      paths = [directory / name for name in ('model.pkl', 'metadata.pkl', f'dm_warp_{width}x{height}_tinygrad.pkl')]
      for path, obj in zip(paths, (run, {'input_shapes': {'calib': (1, 3)}, 'output_slices': {'pixels': slice(0, 4)}}, warp), strict=True):
        with path.open('wb') as f:
          pickle.dump(obj, f)
      with patch('openpilot.selfdrive.modeld.dmonitoringmodeld.MODEL_PKL_PATH', paths[0]), \
           patch('openpilot.selfdrive.modeld.dmonitoringmodeld.METADATA_PATH', paths[1]), \
           patch('openpilot.selfdrive.modeld.dmonitoringmodeld.MODELS_DIR', directory), \
           patch('openpilot.selfdrive.modeld.dmonitoringmodeld.get_tg_input_devices', return_value={'DEV': 'CPU'}), \
           patch('tinygrad.engine.jit.link_linear', wraps=link_linear) as link:
        with patch('tinygrad.engine.jit.run_linear', side_effect=AssertionError('inference during initialization')):
          model = DriverMonitoringModelState(width, height)
        self.assertEqual(link.call_count, 2, 'both DM models must be linked before receiving a camera frame')
        np.testing.assert_array_equal(model.numpy_inputs['calib'], 0)
        np.testing.assert_array_equal(model.warp_inputs_np['transform'], 0)

        name = f'test_dm_startup_{uuid.uuid4().hex}'
        server = VisionIpcServer(name)
        server.create_buffers_with_sizes(0, 2, width, height, size, stride, stride * y_height)
        server.start_listener()
        client = VisionIpcClient(name, 0, False)
        self.assertTrue(client.connect(True))
        for index, value in enumerate((16, 64, 128, 235)):
          server.send(0, np.full(size, value, dtype=np.uint8))
          buf = client.recv()
          self.assertIsNotNone(buf)
          output, _ = model.run(buf, np.full(3, index, dtype=np.float32), np.eye(3, dtype=np.float32) * (index + 1))
          np.testing.assert_array_equal(output, np.full(4, value + 6 * index + 3))
          self.assertEqual(link.call_count, 2, 'camera calls must reuse both prepared DM models')
        self.assertEqual(len(model._blob_cache), 2)

  def test_model_is_linked_without_advancing_history(self):
    width, height = 64, 32
    frame_size = nv12_copy_size(*get_nv12_info(width, height)[:3])
    shapes = {'img': (1, 12, 2, 2), 'features_buffer': (1, 2, 4), 'desire_pulse': (1, 3, 8),
              'traffic_convention': (1, 2), 'action_t': (1, 2)}
    metadata = {'input_shapes': shapes, 'output_slices': {'hidden_state': slice(0, 4), 'pixels': slice(4, 8)}}
    frame_skip = ModelConstants.MODEL_RUN_FREQ // ModelConstants.MODEL_CONTEXT_FREQ
    queues, _, _ = make_input_queues(shapes, frame_skip, 'CPU', frame_size, direct_camera_frames=True)

    def warp(tfm, big_tfm, frame, big_frame):
      return frame[:4].cat(big_frame[:4]).cast('float32').reshape(1, 8)

    def policy(warped, img_q, big_img_q, feat_q, desire_q, packed_npy_inputs):
      feat_q.assign(feat_q + 1).realize()
      return feat_q[0, 0].cat(warped[0, 4:]).reshape(1, 8),

    run = TinyJit(make_run_model(warp, policy, metadata, frame_size), prune=True)
    for _ in range(3):
      run(**{k: queues[k] for k in MODELD_INPUTS + CAMERA_INPUTS})[0].numpy()
    artifact = {'input_devices': {'model': 'CPU'}, 'direct_camera_frames': True,
                'metadata': metadata, 'run_model': {(width, height): run}}
    with tempfile.TemporaryDirectory() as tmp:
      path = Path(tmp) / 'model.pkl'
      with path.open('wb') as f:
        dump_oob(artifact, f)
      with patch('openpilot.selfdrive.modeld.modeld.modeld_pkl_path', return_value=path), \
           patch('openpilot.selfdrive.modeld.modeld.Parser', return_value=Parser(ignore_missing=True)), \
           patch('tinygrad.engine.jit.link_linear', wraps=link_linear) as link:
        with patch('tinygrad.engine.jit.run_linear', side_effect=AssertionError('inference during initialization')):
          model = ModelState(width, height, False)
        self.assertEqual(link.call_count, 1, 'model must be linked before the first camera call')
        np.testing.assert_array_equal(model.input_queues['feat_q'].numpy(), 0)
        np.testing.assert_array_equal(model.prev_desire, 0)
        np.testing.assert_array_equal(model.npy['prev_feat'], 0)

        for count, value in enumerate((16, 64, 128, 235), 1):
          frame = np.full(frame_size, value, dtype=np.uint8)
          inputs = {'desire_pulse': np.zeros(8, dtype=np.float32),
                    'traffic_convention': np.zeros(2, dtype=np.float32), 'action_t': np.zeros(2, dtype=np.float32)}
          outputs = model.run({'img': frame, 'big_img': frame}, dict.fromkeys(('img', 'big_img'), np.eye(3)), inputs)
          np.testing.assert_array_equal(outputs['pixels'], np.full((1, 4), value))
          np.testing.assert_array_equal(outputs['hidden_state'], np.full((1, 4), count))
          np.testing.assert_array_equal(model.input_queues['feat_q'].numpy(), count)
          np.testing.assert_array_equal(model.npy['prev_feat'], np.full((1, 4), count))
          self.assertEqual(link.call_count, 1, 'camera calls must reuse the prepared model')
