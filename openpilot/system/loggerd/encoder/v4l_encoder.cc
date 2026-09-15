#include <cassert>
#include <cstdlib>
#include <string>
#include <sys/ioctl.h>
#include <poll.h>
#include <utility>

#include "system/loggerd/encoder/v4l_encoder.h"
#include "common/util.h"
#include "common/timing.h"

#include <media/msm_media_info.h>

// has to be in this order
#include <linux/v4l2-controls.h>
#include <linux/videodev2.h>
#define V4L2_QCOM_BUF_FLAG_CODECCONFIG 0x00020000
#define V4L2_QCOM_BUF_FLAG_EOS 0x02000000

// AGNOS builds use downstream headers that predate the standard HEVC controls.
#ifndef V4L2_CID_MPEG_VIDEO_HEVC_PROFILE
#define V4L2_CID_MPEG_VIDEO_HEVC_PROFILE (V4L2_CID_MPEG_BASE + 615)
#define V4L2_CID_MPEG_VIDEO_HEVC_LEVEL (V4L2_CID_MPEG_BASE + 616)
#define V4L2_MPEG_VIDEO_HEVC_PROFILE_MAIN 0
#define V4L2_MPEG_VIDEO_HEVC_LEVEL_5 7
#endif

/*
  kernel debugging:
  echo 0xff > /sys/module/videobuf2_core/parameters/debug
  echo 0x7fffffff > /sys/kernel/debug/msm_vidc/debug_level
  echo 0xff > /sys/devices/platform/soc/aa00000.qcom,vidc/video4linux/video33/dev_debug
*/
const int env_debug_encoder = (getenv("DEBUG_ENCODER") != NULL) ? atoi(getenv("DEBUG_ENCODER")) : 0;

static void dequeue_buffer(int fd, v4l2_buf_type buf_type, bool dmabuf, unsigned int *index=NULL, unsigned int *bytesused=NULL, unsigned int *flags=NULL, struct timeval *timestamp=NULL, unsigned int *offset=NULL) {
  v4l2_plane plane = {0};
  v4l2_buffer v4l_buf = {
    .type = buf_type,
    .memory = dmabuf ? V4L2_MEMORY_DMABUF : V4L2_MEMORY_USERPTR,
    .m = { .planes = &plane, },
    .length = 1,
  };
  util::safe_ioctl(fd, VIDIOC_DQBUF, &v4l_buf, "VIDIOC_DQBUF failed");

  if (index) *index = v4l_buf.index;
  if (bytesused) *bytesused = v4l_buf.m.planes[0].bytesused;
  if (flags) *flags = v4l_buf.flags;
  if (timestamp) *timestamp = v4l_buf.timestamp;
  if (offset) *offset = plane.data_offset;
  else assert(plane.data_offset == 0);
}

static void queue_buffer(int fd, v4l2_buf_type buf_type, bool dmabuf, unsigned int index, VisionBuf *buf, struct timeval timestamp={}) {
  v4l2_plane plane = {
    .bytesused = dmabuf && buf_type == V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE ? 0 : (uint32_t)buf->len,
    .length = (unsigned int)buf->len,
    .m = { .userptr = (unsigned long)buf->addr, },
    .reserved = {(unsigned int)buf->fd}
  };
  if (dmabuf) {
    plane.m.fd = buf->fd;
    plane.reserved[0] = 0;
  }

  v4l2_buffer v4l_buf = {
    .index = index,
    .type = buf_type,
    .flags = V4L2_BUF_FLAG_TIMESTAMP_COPY,
    .timestamp = timestamp,
    .memory = dmabuf ? V4L2_MEMORY_DMABUF : V4L2_MEMORY_USERPTR,
    .m = { .planes = &plane, },
    .length = 1,
  };
  util::safe_ioctl(fd, VIDIOC_QBUF, &v4l_buf, "VIDIOC_QBUF failed");
}

static void request_buffers(int fd, v4l2_buf_type buf_type, bool dmabuf, unsigned int count) {
  struct v4l2_requestbuffers reqbuf = {
    .count = count,
    .type = buf_type,
    .memory = dmabuf ? V4L2_MEMORY_DMABUF : V4L2_MEMORY_USERPTR,
  };
  util::safe_ioctl(fd, VIDIOC_REQBUFS, &reqbuf, "VIDIOC_REQBUFS failed");
  assert(reqbuf.count >= count);
}

// Venus joins codec headers to the first slice. Keep the existing callback/message contract.
static size_t annexb_header_size(const uint8_t *data, size_t size, bool hevc) {
  for (size_t i = 0; i + 3 < size; ++i) {
    if (data[i] || data[i + 1]) continue;
    size_t prefix = data[i + 2] == 1 ? 3 : (data[i + 2] == 0 && data[i + 3] == 1 ? 4 : 0);
    if (!prefix || i + prefix >= size) continue;
    unsigned int type = hevc ? (data[i + prefix] >> 1) & 0x3f : data[i + prefix] & 0x1f;
    if (hevc ? type <= 31 : type >= 1 && type <= 5) return i;
  }
  return size;
}

// The mainline codec does not scale. Only reduced-size encodes need a CPU copy.
static void scale_nv12(VisionBuf *src, VisionBuf *dst) {
  src->begin_cpu_access();
  dst->begin_cpu_access();
  for (size_t y = 0; y < dst->height; ++y) {
    const uint8_t *row = src->y + (y * src->height / dst->height) * src->stride;
    for (size_t x = 0; x < dst->width; ++x) dst->y[y * dst->stride + x] = row[x * src->width / dst->width];
  }
  for (size_t y = 0; y < dst->height / 2; ++y) {
    const uint8_t *row = src->uv + (y * src->height / dst->height) * src->stride;
    for (size_t x = 0; x < dst->width / 2; ++x) {
      size_t sx = (x * src->width / dst->width) * 2;
      dst->uv[y * dst->stride + 2 * x] = row[sx];
      dst->uv[y * dst->stride + 2 * x + 1] = row[sx + 1];
    }
  }
  dst->end_cpu_access();
  src->end_cpu_access();
}

void V4LEncoder::dequeue_handler(V4LEncoder *e) {
  std::string dequeue_thread_name = "dq-"+std::string(e->encoder_info.publish_name);
  util::set_thread_name(dequeue_thread_name.c_str());

  e->segment_num++;
  uint32_t idx = -1;
  bool exit = false;

  // POLLIN is capture, POLLOUT is frame. Qualcomm's reference client also
  // requests the corresponding normal-data bits.
  struct pollfd pfd;
  pfd.events = POLLIN | POLLRDNORM | POLLOUT | POLLWRNORM;
  pfd.fd = e->fd;

  // save the header
  kj::Array<capnp::byte> header;

  while (!exit) {
    int rc = poll(&pfd, 1, 1000);
    if (rc < 0) {
      if (errno != EINTR) {
        // TODO: exit encoder?
        // ignore the error and keep going
        LOGE("poll failed (%d - %d)", rc, errno);
      }
      continue;
    } else if (rc == 0) {
      LOGE("encoder dequeue poll timeout");
      continue;
    }

    if (env_debug_encoder >= 2) {
      printf("%20s poll %x at %.2f ms\n", e->encoder_info.publish_name, pfd.revents, millis_since_boot());
    }

    if (e->venus && (pfd.revents & (POLLERR | POLLHUP | POLLNVAL))) {
      LOGE("encoder %s poll error: %x", e->encoder_info.publish_name, pfd.revents);
      std::abort();
    }

    int frame_id = -1;
    if (pfd.revents & (POLLIN | POLLRDNORM)) {
      unsigned int bytesused, flags, index, offset;
      struct timeval timestamp;
      dequeue_buffer(e->fd, V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE, e->venus, &index, &bytesused, &flags, &timestamp, &offset);
      assert(offset <= bytesused && bytesused <= e->buf_out[index].len);
      e->buf_out[index].begin_cpu_access();
      uint8_t *buf = (uint8_t*)e->buf_out[index].addr + offset;
      bytesused -= offset;
      int64_t ts = timestamp.tv_sec * 1000000 + timestamp.tv_usec;

      if (e->venus && bytesused && (flags & V4L2_BUF_FLAG_KEYFRAME)) {
        size_t header_size = annexb_header_size(buf, bytesused, e->is_h265);
        assert(header_size < bytesused);
        if (header_size) {
          bool first_header = header.size() == 0;
          header = kj::heapArray<capnp::byte>(buf, header_size);
          if (first_header && e->packet_callback) e->packet_callback(header.begin(), header.size(), ts, true, false);
        }
        assert(header.size() > 0);
        buf += header_size;
        bytesused -= header_size;
      }

      // eof packet, we exit
      if ((e->venus && !bytesused && (flags & V4L2_BUF_FLAG_LAST)) || (!e->venus && (flags & V4L2_QCOM_BUF_FLAG_EOS))) {
        exit = true;
      } else if (flags & V4L2_QCOM_BUF_FLAG_CODECCONFIG) {
        // save header
        header = kj::heapArray<capnp::byte>(buf, bytesused);
        if (e->packet_callback) e->packet_callback(header.begin(), header.size(), ts, true, false);
      } else {
        VisionIpcBufExtra extra = e->extras.pop();
        assert(extra.timestamp_eof/1000 == ts); // stay in sync
        frame_id = extra.frame_id;
        ++idx;
        if (e->packet_callback) {
          e->packet_callback(buf, bytesused, ts, false, flags & V4L2_BUF_FLAG_KEYFRAME);
        } else {
          e->publisher_publish(e->segment_num, idx, extra, flags, header, kj::arrayPtr<capnp::byte>(buf, bytesused));
        }
      }

      e->buf_out[index].end_cpu_access();
      if (env_debug_encoder) {
        printf("%20s got(%d) %6d bytes flags %8x idx %3d/%4d id %8d ts %ld lat %.2f ms (%lu frames free)\n",
          e->encoder_info.publish_name, index, bytesused, flags, e->segment_num, idx, frame_id, ts, millis_since_boot()-(ts/1000.), e->free_buf_in.size());
      }

      // requeue the buffer
      if (e->venus && (flags & V4L2_BUF_FLAG_LAST)) {
        exit = true;
        e->drained_buf = index;
      } else {
        queue_buffer(e->fd, V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE, e->venus, index, &e->buf_out[index]);
      }
    }

    if (pfd.revents & (POLLOUT | POLLWRNORM)) {
      unsigned int index;
      dequeue_buffer(e->fd, V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE, e->venus, &index);
      VisionBuf *input_buf = e->input_bufs[index].exchange(nullptr);
      if (input_buf && e->input_done_callback) e->input_done_callback(input_buf);
      e->free_buf_in.push(index);
    }
  }
}

V4LEncoder::V4LEncoder(const EncoderInfo &encoder_info, int in_width, int in_height)
    : V4LEncoder(encoder_info, in_width, in_height, Options{}) {}

V4LEncoder::V4LEncoder(const EncoderInfo &encoder_info, int in_width, int in_height, Options options)
    : VideoEncoder(encoder_info, in_width, in_height), packet_callback(std::move(options.packet_callback)),
      input_done_callback(std::move(options.input_done_callback)) {
  const char *paths[] = {
    "/dev/v4l/by-path/platform-aa00000.qcom_vidc-video-index1",
    "/dev/v4l/by-path/platform-aa00000.video-codec-video-index1",
    "/dev/v4l/by-path/platform-aa00000.video-codec-video-index0",
  };
  struct v4l2_capability cap = {};
  for (const char *path : paths) {
    fd = HANDLE_EINTR(open(path, O_RDWR|O_NONBLOCK));
    if (fd < 0) {
      if (errno == ENOENT) continue;
      break;
    }
    util::safe_ioctl(fd, VIDIOC_QUERYCAP, &cap, "VIDIOC_QUERYCAP failed");
    // Venus encoder/decoder indices depend on probe order.
    if ((strcmp((const char *)cap.driver, "qcom-venus") == 0 && strcmp((const char *)cap.card, "Qualcomm Venus video encoder") == 0) ||
        (strcmp((const char *)cap.driver, "msm_vidc_driver") == 0 && strcmp((const char *)cap.card, "msm_vidc_venc") == 0)) break;
    close(fd);
    fd = -1;
  }
  assert(fd >= 0);
  LOGD("opened encoder device %s %s = %d", cap.driver, cap.card, fd);
  venus = strcmp((const char *)cap.driver, "qcom-venus") == 0;

  EncoderSettings encoder_settings = encoder_info.get_settings(in_width);
  current_bitrate = encoder_settings.bitrate;
  is_h265 = encoder_settings.encode_type == cereal::EncodeIndex::Type::FULL_H_E_V_C;

  struct v4l2_format fmt_out = {
    .type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE,
    .fmt = {
      .pix_mp = {
        // downscales are free with v4l
        .width = (unsigned int)(out_width),
        .height = (unsigned int)(out_height),
        .pixelformat = is_h265 ? V4L2_PIX_FMT_HEVC : V4L2_PIX_FMT_H264,
        .field = V4L2_FIELD_ANY,
        .colorspace = V4L2_COLORSPACE_DEFAULT,
      }
    }
  };
  util::safe_ioctl(fd, VIDIOC_S_FMT, &fmt_out, "VIDIOC_S_FMT failed");

  v4l2_streamparm streamparm = {
    .type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE,
    .parm = {
      .output = {
        // TODO: more stuff here? we don't know
        .timeperframe = {
          .numerator = 1,
          .denominator = (unsigned int)encoder_info.fps
        }
      }
    }
  };
  util::safe_ioctl(fd, VIDIOC_S_PARM, &streamparm, "VIDIOC_S_PARM failed");

  struct v4l2_format fmt_in = {
    .type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE,
    .fmt = {
      .pix_mp = {
        .width = (unsigned int)(venus ? out_width : in_width),
        .height = (unsigned int)(venus ? out_height : in_height),
        .pixelformat = options.input_format,
        .field = V4L2_FIELD_ANY,
        .colorspace = V4L2_COLORSPACE_470_SYSTEM_BG,
      }
    }
  };
  util::safe_ioctl(fd, VIDIOC_S_FMT, &fmt_in, "VIDIOC_S_FMT failed");

  LOGD("in buffer size %d, out buffer size %d",
    fmt_in.fmt.pix_mp.plane_fmt[0].sizeimage,
    fmt_out.fmt.pix_mp.plane_fmt[0].sizeimage);

  if (venus) {
    assert(options.input_format == V4L2_PIX_FMT_NV12);
    struct v4l2_selection crop = {
      .type = V4L2_BUF_TYPE_VIDEO_OUTPUT,
      .target = V4L2_SEL_TGT_CROP,
      .r = {.width = (unsigned int)out_width, .height = (unsigned int)out_height},
    };
    util::safe_ioctl(fd, VIDIOC_S_SELECTION, &crop, "VIDIOC_S_SELECTION failed");
    assert(crop.r.width == out_width && crop.r.height == out_height);

    struct v4l2_control ctrls[] = {
      {.id = V4L2_CID_MPEG_VIDEO_BITRATE, .value = encoder_settings.bitrate},
      {.id = V4L2_CID_MPEG_VIDEO_BITRATE_MODE, .value = V4L2_MPEG_VIDEO_BITRATE_MODE_VBR},
      {.id = V4L2_CID_MPEG_VIDEO_GOP_SIZE, .value = encoder_settings.gop_size},
      {.id = V4L2_CID_MPEG_VIDEO_B_FRAMES, .value = encoder_settings.b_frames},
      {.id = V4L2_CID_MPEG_VIDEO_HEADER_MODE, .value = V4L2_MPEG_VIDEO_HEADER_MODE_JOINED_WITH_1ST_FRAME},
      {.id = static_cast<uint32_t>(is_h265 ? V4L2_CID_MPEG_VIDEO_HEVC_PROFILE : V4L2_CID_MPEG_VIDEO_H264_PROFILE),
       .value = is_h265 ? V4L2_MPEG_VIDEO_HEVC_PROFILE_MAIN : V4L2_MPEG_VIDEO_H264_PROFILE_HIGH},
      {.id = static_cast<uint32_t>(is_h265 ? V4L2_CID_MPEG_VIDEO_HEVC_LEVEL : V4L2_CID_MPEG_VIDEO_H264_LEVEL),
       .value = is_h265 ? V4L2_MPEG_VIDEO_HEVC_LEVEL_5 : V4L2_MPEG_VIDEO_H264_LEVEL_3_1},
    };
    for (auto ctrl : ctrls) util::safe_ioctl(fd, VIDIOC_S_CTRL, &ctrl, "VIDIOC_S_CTRL failed");

    if (out_width != in_width || out_height != in_height) {
      for (auto &buf : scaled_bufs) {
        buf.allocate(fmt_in.fmt.pix_mp.plane_fmt[0].sizeimage);
        buf.init_yuv(out_width, out_height, fmt_in.fmt.pix_mp.plane_fmt[0].bytesperline,
                     fmt_in.fmt.pix_mp.plane_fmt[0].bytesperline * fmt_in.fmt.pix_mp.height);
      }
    }
  } else {
    // shared ctrls
    {
      struct v4l2_control ctrls[] = {
        { .id = V4L2_CID_MPEG_VIDEO_BITRATE, .value = encoder_settings.bitrate},
        { .id = V4L2_CID_MPEG_VIDC_VIDEO_NUM_P_FRAMES, .value = encoder_settings.gop_size - encoder_settings.b_frames - 1},
        { .id = V4L2_CID_MPEG_VIDC_VIDEO_NUM_B_FRAMES, .value = encoder_settings.b_frames},
        { .id = V4L2_CID_MPEG_VIDEO_HEADER_MODE, .value = V4L2_MPEG_VIDEO_HEADER_MODE_SEPARATE},
        { .id = V4L2_CID_MPEG_VIDC_VIDEO_RATE_CONTROL, .value = V4L2_CID_MPEG_VIDC_VIDEO_RATE_CONTROL_VBR_CFR},
        { .id = V4L2_CID_MPEG_VIDC_VIDEO_PRIORITY, .value = V4L2_MPEG_VIDC_VIDEO_PRIORITY_REALTIME_DISABLE},
        { .id = V4L2_CID_MPEG_VIDC_VIDEO_IDR_PERIOD, .value = 1},
      };
      for (auto ctrl : ctrls) {
        util::safe_ioctl(fd, VIDIOC_S_CTRL, &ctrl, "VIDIOC_S_CTRL failed");
      }
    }
    if (options.max_performance) {
      struct v4l2_control ctrl = {
        .id = V4L2_CID_MPEG_VIDC_VIDEO_PRIORITY,
        .value = V4L2_MPEG_VIDC_VIDEO_PRIORITY_REALTIME_ENABLE,
      };
      util::safe_ioctl(fd, VIDIOC_S_CTRL, &ctrl, "VIDIOC_S_CTRL offline encode failed");
    }

    if (is_h265) {
      struct v4l2_control ctrls[] = {
        { .id = V4L2_CID_MPEG_VIDC_VIDEO_HEVC_PROFILE, .value = V4L2_MPEG_VIDC_VIDEO_HEVC_PROFILE_MAIN},
        { .id = V4L2_CID_MPEG_VIDC_VIDEO_HEVC_TIER_LEVEL, .value = V4L2_MPEG_VIDC_VIDEO_HEVC_LEVEL_HIGH_TIER_LEVEL_5},
        { .id = V4L2_CID_MPEG_VIDC_VIDEO_VUI_TIMING_INFO, .value = V4L2_MPEG_VIDC_VIDEO_VUI_TIMING_INFO_ENABLED},
      };
      for (auto ctrl : ctrls) {
        util::safe_ioctl(fd, VIDIOC_S_CTRL, &ctrl, "VIDIOC_S_CTRL failed");
      }
    } else {
      if (encoder_info.is_live) {
        struct v4l2_control ctrls[] = {
          { .id = V4L2_CID_MPEG_VIDEO_H264_PROFILE, .value = V4L2_MPEG_VIDEO_H264_PROFILE_HIGH},
          { .id = V4L2_CID_MPEG_VIDEO_H264_LEVEL, .value = V4L2_MPEG_VIDEO_H264_LEVEL_3_1},
          { .id = V4L2_CID_MPEG_VIDEO_H264_ENTROPY_MODE, .value = V4L2_MPEG_VIDEO_H264_ENTROPY_MODE_CABAC},
          { .id = V4L2_CID_MPEG_VIDC_VIDEO_H264_CABAC_MODEL, .value = V4L2_CID_MPEG_VIDC_VIDEO_H264_CABAC_MODEL_0},
        };
        for (auto ctrl : ctrls) {
          util::safe_ioctl(fd, VIDIOC_S_CTRL, &ctrl, "VIDIOC_S_CTRL failed");
        }
      } else {
        struct v4l2_control ctrls[] = {
          { .id = V4L2_CID_MPEG_VIDEO_H264_PROFILE, .value = V4L2_MPEG_VIDEO_H264_PROFILE_HIGH},
          { .id = V4L2_CID_MPEG_VIDEO_H264_LEVEL, .value = V4L2_MPEG_VIDEO_H264_LEVEL_UNKNOWN},
          { .id = V4L2_CID_MPEG_VIDEO_H264_ENTROPY_MODE, .value = V4L2_MPEG_VIDEO_H264_ENTROPY_MODE_CABAC},
          { .id = V4L2_CID_MPEG_VIDC_VIDEO_H264_CABAC_MODEL, .value = V4L2_CID_MPEG_VIDC_VIDEO_H264_CABAC_MODEL_0},
        };
        for (auto ctrl : ctrls) {
          util::safe_ioctl(fd, VIDIOC_S_CTRL, &ctrl, "VIDIOC_S_CTRL failed");
        }
      }

      struct v4l2_control ctrls[] = {
        { .id = V4L2_CID_MPEG_VIDEO_H264_LOOP_FILTER_MODE, .value = V4L2_MPEG_VIDEO_H264_LOOP_FILTER_MODE_ENABLED},
        { .id = V4L2_CID_MPEG_VIDEO_H264_LOOP_FILTER_ALPHA, .value = 0},
        { .id = V4L2_CID_MPEG_VIDEO_H264_LOOP_FILTER_BETA, .value = 0},
        { .id = V4L2_CID_MPEG_VIDEO_MULTI_SLICE_MODE, .value = 0},
      };
      for (auto ctrl : ctrls) {
        util::safe_ioctl(fd, VIDIOC_S_CTRL, &ctrl, "VIDIOC_S_CTRL failed");
      }
    }

  }

  // allocate buffers
  request_buffers(fd, V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE, venus, BUF_OUT_COUNT);
  request_buffers(fd, V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE, venus, BUF_IN_COUNT);
  if (venus) util::safe_ioctl(fd, VIDIOC_G_FMT, &fmt_out, "VIDIOC_G_FMT failed");

  // start encoder
  v4l2_buf_type buf_type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  util::safe_ioctl(fd, VIDIOC_STREAMON, &buf_type, "VIDIOC_STREAMON failed");
  buf_type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
  util::safe_ioctl(fd, VIDIOC_STREAMON, &buf_type, "VIDIOC_STREAMON failed");

  // queue up output buffers
  for (unsigned int i = 0; i < BUF_OUT_COUNT; i++) {
    buf_out[i].allocate(fmt_out.fmt.pix_mp.plane_fmt[0].sizeimage);
    queue_buffer(fd, V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE, venus, i, &buf_out[i]);
  }
  // queue up input buffers
  for (unsigned int i = 0; i < BUF_IN_COUNT; i++) {
    free_buf_in.push(i);
  }
}

void V4LEncoder::encoder_open() {
  if (venus && segment_num >= 0) {
    struct v4l2_encoder_cmd cmd = {.cmd = V4L2_ENC_CMD_START};
    util::safe_ioctl(fd, VIDIOC_ENCODER_CMD, &cmd, "VIDIOC_ENCODER_CMD start failed");
    // Venus returns empty buffers if they are queued while the session is stopped.
    assert(drained_buf >= 0);
    queue_buffer(fd, V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE, venus, drained_buf, &buf_out[drained_buf]);
    drained_buf = -1;
    request_keyframe();
  }
  dequeue_handler_thread = std::thread(V4LEncoder::dequeue_handler, this);
  this->is_open = true;
  this->counter = 0;
}

int V4LEncoder::encode_frame(VisionBuf* buf, VisionIpcBufExtra *extra) {
  struct timeval timestamp {
    .tv_sec = (long)(extra->timestamp_eof/1000000000),
    .tv_usec = (long)((extra->timestamp_eof/1000) % 1000000),
  };

  // reserve buffer
  int buffer_in = free_buf_in.pop();
  input_bufs[buffer_in].store(buf);

  // push buffer
  extras.push(*extra);
  //buf->sync(VISIONBUF_SYNC_TO_DEVICE);
  VisionBuf *input = buf;
  if (venus && scaled_bufs[buffer_in].len) {
    input = &scaled_bufs[buffer_in];
    scale_nv12(buf, input);
  }
  queue_buffer(fd, V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE, venus, buffer_in, input, timestamp);

  return this->counter++;
}

void V4LEncoder::encoder_close() {
  if (this->is_open) {
    // pop all the frames before closing, then put the buffers back
    for (int i = 0; i < BUF_IN_COUNT; i++) free_buf_in.pop();
    for (int i = 0; i < BUF_IN_COUNT; i++) free_buf_in.push(i);
    // no frames, stop the encoder
    struct v4l2_encoder_cmd encoder_cmd = { .cmd = V4L2_ENC_CMD_STOP };
    util::safe_ioctl(fd, VIDIOC_ENCODER_CMD, &encoder_cmd, "VIDIOC_ENCODER_CMD failed");
    // Wait for the last encoded frame and the driver's end-of-stream buffer.
    dequeue_handler_thread.join();
    assert(extras.empty());
  }
  this->is_open = false;
}

void V4LEncoder::set_bitrate(int bitrate) {
  if (bitrate == current_bitrate) return;
  if (bitrate <= 0) {
    LOGE("invalid livestream encoder bitrate %d", bitrate);
    return;
  }

  struct v4l2_control ctrl = {
    .id = V4L2_CID_MPEG_VIDEO_BITRATE,
    .value = bitrate,
  };

  if (util::safe_ioctl(fd, VIDIOC_S_CTRL, &ctrl) == -1) {
    LOGE("failed to update %s bitrate to %d", encoder_info.publish_name, bitrate);
    return;
  }
  current_bitrate = bitrate;
}

void V4LEncoder::request_keyframe() {
  struct v4l2_control ctrl = {
    .id = static_cast<uint32_t>(venus ? V4L2_CID_MPEG_VIDEO_FORCE_KEY_FRAME : V4L2_CID_MPEG_VIDC_VIDEO_REQUEST_IFRAME),
    .value = 1,
  };

  if (util::safe_ioctl(fd, VIDIOC_S_CTRL, &ctrl) == -1) {
    LOGE("failed to request keyframe for %s", encoder_info.publish_name);
  }
}

V4LEncoder::~V4LEncoder() {
  encoder_close();
  v4l2_buf_type buf_type = V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE;
  util::safe_ioctl(fd, VIDIOC_STREAMOFF, &buf_type, "VIDIOC_STREAMOFF failed");
  request_buffers(fd, V4L2_BUF_TYPE_VIDEO_OUTPUT_MPLANE, venus, 0);
  buf_type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  util::safe_ioctl(fd, VIDIOC_STREAMOFF, &buf_type, "VIDIOC_STREAMOFF failed");
  request_buffers(fd, V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE, venus, 0);
  close(fd);

  for (auto &buf : scaled_bufs) {
    if (buf.len) buf.free();
  }
  for (int i = 0; i < BUF_OUT_COUNT; i++) {
    if (buf_out[i].free() != 0) {
      LOGE("Failed to free buffer");
    }
  }
}
