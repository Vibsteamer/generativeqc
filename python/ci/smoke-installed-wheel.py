"""Exercise the installed CUDA wheel on CI workers with or without a GPU."""

import ctypes

from generativeqc import _native
from generativeqc.profiles import DeviceDescriptor

descriptor = DeviceDescriptor()
descriptor.struct_size = ctypes.sizeof(descriptor)
descriptor.abi_version = _native.ABI_VERSION
probe = _native.load_library().generativeqc_cuda_tuning_device
probe.argtypes = [ctypes.c_int32, ctypes.POINTER(DeviceDescriptor)]
probe.restype = ctypes.c_int
print("CUDA tuning device status", probe(0, ctypes.byref(descriptor)))

shell = ctypes.CDLL(str(_native.require_f12_library("focal_shell_i")))
projection = shell.gqc_f12_raw_spherical_projection
projection.argtypes = [ctypes.c_uint, ctypes.POINTER(ctypes.c_double), ctypes.c_size_t]
projection.restype = ctypes.c_int
value = ctypes.c_double()
if projection(0, ctypes.byref(value), 1) != 0 or value.value != 1.0:
    raise RuntimeError("installed F12 shell library failed the s-shell projection")

response = ctypes.CDLL(str(_native.require_f12_library("focal_hf_cabs_response")))
for symbol in (
    "gqc_f12_molecular_metric_inverse_cholesky_v1",
    "gqc_f12_molecular_metric_whitener",
    "generativeqc_f12_cabs_singles_cuda_v1",
):
    getattr(response, symbol)
