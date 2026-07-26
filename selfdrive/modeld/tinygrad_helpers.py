from tinygrad.tensor import Tensor

from msgq.visionipc import VisionBuf


def visionbuf_to_tensor(buf: VisionBuf, size: int) -> Tensor:
  ptr = buf.data.ctypes.data
  return Tensor.from_blob(ptr, (size,), dtype='uint8', fd=buf.fd)
