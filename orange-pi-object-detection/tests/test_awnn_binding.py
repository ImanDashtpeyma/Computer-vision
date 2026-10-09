"""Checks the ctypes binding against a tiny C stand-in for libawnn_npu.so.

The real library needs the Allwinner driver and only exists on the board. This
stub exports the same seven functions, so we can verify argument passing,
pointer handling and output copying -- the parts of src/awnn.py that are easy
to get subtly wrong -- on any machine with a C compiler.
"""
from __future__ import annotations

import shutil
import subprocess

import numpy as np
import pytest

from src.awnn import AwnnLibrary

STUB_C = r"""
#include <stdlib.h>
#include <string.h>
static unsigned char last_input_first_byte = 0;
static float out0[4] = {1.f, 2.f, 3.f, 4.f};
static float out1[2] = {0.f, 0.f};
void awnn_init(void) {}
void awnn_uninit(void) {}
void *awnn_create(const char *nbg) {
    return (strlen(nbg) > 0 && nbg[0] != '!') ? (void *)0x1 : NULL;
}
void awnn_destroy(void *ctx) {}
void awnn_set_input_buffers(void *ctx, void **bufs) {
    unsigned char *in = (unsigned char *)bufs[0];
    last_input_first_byte = in[0];
}
void awnn_run(void *ctx) { out1[0] = (float)last_input_first_byte; out1[1] = 42.f; }
float *awnn_get_output_buffer(void *ctx, int i) { return i == 0 ? out0 : out1; }
"""


@pytest.fixture(scope="module")
def stub_lib(tmp_path_factory):
    if shutil.which("gcc") is None:
        pytest.skip("gcc not available")
    d = tmp_path_factory.mktemp("stub")
    src, out = d / "stub.c", d / "libstub.so"
    src.write_text(STUB_C)
    subprocess.run(["gcc", "-shared", "-fPIC", "-o", str(out), str(src)], check=True)
    return AwnnLibrary(out)


def test_run_passes_the_input_and_returns_copied_outputs(stub_lib):
    ctx = stub_lib.create("model.nb")
    inp = np.full((3, 4, 4), 7, dtype=np.uint8)

    outputs = stub_lib.run(ctx, inp, [4, 2])

    assert outputs[0].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert outputs[1].tolist() == [7.0, 42.0]  # 7 = first input byte seen by the C side
    assert outputs[0].dtype == np.float32


def test_non_contiguous_input_is_made_contiguous(stub_lib):
    ctx = stub_lib.create("model.nb")
    base = np.zeros((4, 4, 3), dtype=np.uint8)
    base[0, 0, 0] = 9
    outputs = stub_lib.run(ctx, base.transpose(2, 0, 1), [4, 2])  # a non-contiguous view
    assert outputs[1][0] == 9.0


def test_create_failure_raises_a_clear_error(stub_lib):
    with pytest.raises(RuntimeError, match="awnn_create failed"):
        stub_lib.create("!bad")


def test_missing_library_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="build_awnn.sh"):
        AwnnLibrary(tmp_path / "nope.so")
