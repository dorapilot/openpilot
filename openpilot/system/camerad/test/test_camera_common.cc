#include <algorithm>

#include "common/tests/native_test.h"
#include "system/camerad/cameras/camera_common.h"

void test_exposure_stride() {
  constexpr int width = 6, height = 4;
  for (const int stride : {width, 16}) {
    VisionBuf buf;
    buf.allocate(stride * height * 3 / 2);
    buf.init_yuv(width, height, stride, stride * height);
    buf.begin_cpu_access(true);
    std::fill_n(buf.y, buf.len, 0);
    for (int y = 0; y < height; y++) {
      std::fill_n(buf.y + y * stride, width, 192);
    }
    buf.end_cpu_access(true);
    CameraBuf camera{};
    camera.cur_yuv_buf = &buf;
    camera.out_img_width = width;
    camera.out_img_height = height;
    CHECK(calculate_exposure_value(&camera, {1, 1, 3, 2}, 1, 1) == 192.0f / 256);
    CHECK(buf.free() == 0);
  }
}

int main() {
  return run_native_test(test_exposure_stride);
}
