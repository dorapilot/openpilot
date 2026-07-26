from types import SimpleNamespace

import numpy as np

from tinygrad import Context

from openpilot.selfdrive.modeld.tinygrad_helpers import visionbuf_to_tensor


def test_visionbuf_to_tensor_preserves_dma_buf_fd():
  data = np.arange(16, dtype=np.uint8)
  visionbuf = SimpleNamespace(data=data, fd=7)

  with Context(DEV="CPU"):
    tensor = visionbuf_to_tensor(visionbuf, data.size)

  np.testing.assert_array_equal(tensor.numpy(), data)
  buffer = tensor.uop.base.buffer
  assert buffer is not None and buffer.options is not None
  assert buffer.options.external_fd == visionbuf.fd
