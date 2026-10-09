"""ctypes binding for Allwinner's `awnn` helper library (libawnn_npu.so).

`awnn_lib.c` ships in Allwinner's ai-sdk and wraps the VIPLite driver API into
seven simple calls. This repo does not redistribute it: native/build_awnn.sh
compiles it from your local copy of the SDK on the board. See the README.
"""
from __future__ import annotations

import ctypes
import pathlib
import threading

import numpy as np


class AwnnLibrary:
    def __init__(self, lib_path: str | pathlib.Path):
        path = pathlib.Path(lib_path)
        if not path.exists():
            raise FileNotFoundError(
                f"NPU helper library not found at {path}. Build it on the Orange Pi with "
                "`native/build_awnn.sh /path/to/ai-sdk` (see README, 'NPU backend')."
            )
        self._lib = ctypes.CDLL(str(path))
        lib = self._lib
        lib.awnn_init.argtypes = []
        lib.awnn_init.restype = None
        lib.awnn_uninit.argtypes = []
        lib.awnn_uninit.restype = None
        lib.awnn_create.argtypes = [ctypes.c_char_p]
        lib.awnn_create.restype = ctypes.c_void_p
        lib.awnn_destroy.argtypes = [ctypes.c_void_p]
        lib.awnn_destroy.restype = None
        lib.awnn_set_input_buffers.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        lib.awnn_set_input_buffers.restype = None
        lib.awnn_run.argtypes = [ctypes.c_void_p]
        lib.awnn_run.restype = None
        lib.awnn_get_output_buffer.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.awnn_get_output_buffer.restype = ctypes.POINTER(ctypes.c_float)
        self._lock = threading.Lock()

    def init(self) -> None:
        self._lib.awnn_init()

    def uninit(self) -> None:
        self._lib.awnn_uninit()

    def create(self, nbg_path: str) -> int:
        ctx = self._lib.awnn_create(nbg_path.encode())
        if not ctx:
            raise RuntimeError(
                f"awnn_create failed for {nbg_path!r}. Check that the .nb file matches this "
                "chip's NPU version (A733 = v3) and that the VIPLite driver is loaded."
            )
        return ctx

    def destroy(self, ctx: int) -> None:
        self._lib.awnn_destroy(ctx)

    def run(self, ctx: int, input_chw: np.ndarray, output_sizes: list[int]) -> list[np.ndarray]:
        """Run one inference. `input_chw` must be a contiguous uint8 array."""
        if not input_chw.flags["C_CONTIGUOUS"]:
            input_chw = np.ascontiguousarray(input_chw)
        buffers = (ctypes.c_void_p * 1)(input_chw.ctypes.data)
        with self._lock:
            self._lib.awnn_set_input_buffers(ctx, buffers)
            self._lib.awnn_run(ctx)
            outputs = []
            for i, n in enumerate(output_sizes):
                ptr = self._lib.awnn_get_output_buffer(ctx, i)
                outputs.append(np.ctypeslib.as_array(ptr, shape=(n,)).copy())
        return outputs
