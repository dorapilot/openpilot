import os
import glob as globmod
import fcntl
import ctypes
from functools import cache

@cache
def get_tlmm_base() -> int:
  """Discover TLMM GPIO chip base. Non-zero on mainline kernel, 0 on downstream."""
  bases = globmod.glob("/sys/bus/platform/devices/3400000.pinctrl/gpio/*/base")
  if bases:
    try:
      with open(bases[0]) as f:
        return int(f.read().strip())
    except (OSError, ValueError):
      pass
  return 0

def _sysfs_pin(pin: int) -> int:
  return get_tlmm_base() + pin

def gpio_init(pin: int, output: bool) -> None:
  sysfs_pin = _sysfs_pin(pin)
  try:
    with open(f"/sys/class/gpio/gpio{sysfs_pin}/direction", 'wb') as f:
      f.write(b"out" if output else b"in")
  except Exception as e:
    print(f"Failed to set gpio {pin} direction: {e}")

def gpio_set(pin: int, high: bool) -> None:
  sysfs_pin = _sysfs_pin(pin)
  try:
    with open(f"/sys/class/gpio/gpio{sysfs_pin}/value", 'wb') as f:
      f.write(b"1" if high else b"0")
  except Exception as e:
    print(f"Failed to set gpio {pin} value: {e}")

def gpio_read(pin: int) -> bool | None:
  sysfs_pin = _sysfs_pin(pin)
  val = None
  try:
    with open(f"/sys/class/gpio/gpio{sysfs_pin}/value", 'rb') as f:
      val = bool(int(f.read().strip()))
  except Exception as e:
    print(f"Failed to read gpio {pin} value: {e}")

  return val

def gpio_export(pin: int) -> None:
  sysfs_pin = _sysfs_pin(pin)
  if os.path.isdir(f"/sys/class/gpio/gpio{sysfs_pin}"):
    return

  try:
    with open("/sys/class/gpio/export", 'w') as f:
      f.write(str(sysfs_pin))
  except Exception:
    print(f"Failed to export gpio {pin}")

@cache
def get_irq_action(irq: int) -> list[str]:
  try:
    with open(f"/sys/kernel/irq/{irq}/actions") as f:
      actions = f.read().strip().split(',')
      return actions
  except FileNotFoundError:
    return []

def get_irqs_for_action(action: str) -> list[str]:
  ret = []
  with open("/proc/interrupts") as f:
    for l in f.readlines():
      irq = l.split(':')[0].strip()
      if irq.isdigit() and action in get_irq_action(irq):
        ret.append(irq)
  return ret

# *** gpiochip ***

def get_tlmm_gpiochip() -> int:
  chips = globmod.glob('/sys/bus/platform/devices/3400000.pinctrl/gpiochip*')
  return int(os.path.basename(chips[0]).removeprefix('gpiochip')) if chips else 0

class gpioevent_data(ctypes.Structure):
  # GPIO v1 and v2 events share this timestamp/id prefix.
  _fields_ = [
    ("timestamp", ctypes.c_uint64),
    ("id", ctypes.c_uint32),
  ]

class gpioevent_request(ctypes.Structure):
  _fields_ = [
    ("lineoffset", ctypes.c_uint32),
    ("handleflags", ctypes.c_uint32),
    ("eventflags", ctypes.c_uint32),
    ("label", ctypes.c_char * 32),
    ("fd", ctypes.c_int)
  ]

class gpio_v2_line_config(ctypes.Structure):
  _fields_ = [
    ("flags", ctypes.c_uint64),
    ("num_attrs", ctypes.c_uint32),
    ("padding", ctypes.c_uint32 * 5),
    # Ten 24-byte attribute slots; this request uses only the default flags.
    ("attrs", ctypes.c_uint64 * 30),
  ]

class gpio_v2_line_request(ctypes.Structure):
  _fields_ = [
    ("offsets", ctypes.c_uint32 * 64),
    ("consumer", ctypes.c_char * 32),
    ("config", gpio_v2_line_config),
    ("num_lines", ctypes.c_uint32),
    ("event_buffer_size", ctypes.c_uint32),
    ("padding", ctypes.c_uint32 * 5),
    ("fd", ctypes.c_int32),
  ]

def gpiochip_get_ro_value_fd(label: str, gpiochip_id: int, pin: int) -> int:
  GPIOEVENT_REQUEST_BOTH_EDGES = 0x3
  GPIOHANDLE_REQUEST_INPUT = 0x1
  GPIO_GET_LINEEVENT_IOCTL = 0xc030b404
  GPIO_V2_LINE_FLAG_INPUT = 0x4
  GPIO_V2_LINE_FLAG_BOTH_EDGES = 0x30
  GPIO_V2_GET_LINE_IOCTL = 0xc250b407

  if tuple(map(int, os.uname().release.split('.')[:2])) >= (5, 10):
    rq = gpio_v2_line_request()
    rq.offsets[0] = pin
    rq.num_lines = 1
    rq.config.flags = GPIO_V2_LINE_FLAG_INPUT | GPIO_V2_LINE_FLAG_BOTH_EDGES
    rq.consumer = label.encode('utf-8')[:31] + b'\0'
    request = GPIO_V2_GET_LINE_IOCTL
  else:
    rq = gpioevent_request()
    rq.lineoffset = pin
    rq.handleflags = GPIOHANDLE_REQUEST_INPUT
    rq.eventflags = GPIOEVENT_REQUEST_BOTH_EDGES
    rq.label = label.encode('utf-8')[:31] + b'\0'
    request = GPIO_GET_LINEEVENT_IOCTL

  fd = os.open(f"/dev/gpiochip{gpiochip_id}", os.O_RDONLY)
  try:
    fcntl.ioctl(fd, request, rq)
  finally:
    os.close(fd)
  return int(rq.fd)
