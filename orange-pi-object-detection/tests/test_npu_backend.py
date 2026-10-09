"""NPUBackend tests using a fake awnn library -- no NPU, driver or .nb needed."""
from __future__ import annotations

import numpy as np
import pytest

from src.backends import NPUBackend
from src.config import ModelConfig
from src.yolov5 import NUM_ATTRS


class FakeAwnn:
    def __init__(self):
        self.calls = []
        self.last_input = None

    def init(self):
        self.calls.append("init")

    def create(self, path):
        self.calls.append(f"create:{path}")
        return 1234

    def run(self, ctx, input_chw, output_sizes):
        self.calls.append("run")
        self.last_input = input_chw
        outputs = [np.zeros(n, dtype=np.float32) for n in output_sizes]
        feat = outputs[2].reshape(3, 20, 20, NUM_ATTRS)
        feat[1, 10, 5, 4] = 8.0
        feat[1, 10, 5, 5] = 8.0  # person
        return outputs

    def destroy(self, ctx):
        self.calls.append("destroy")

    def uninit(self):
        self.calls.append("uninit")


@pytest.fixture
def nbg_file(tmp_path):
    path = tmp_path / "yolov5.nb"
    path.write_bytes(b"fake")
    return str(path)


def test_detect_runs_the_network_and_decodes_boxes(nbg_file):
    fake = FakeAwnn()
    backend = NPUBackend(ModelConfig(backend="npu", nbg_path=nbg_file), lib=fake)

    detections = backend.detect(np.zeros((640, 640, 3), dtype=np.uint8))

    assert [d.label for d in detections] == ["person"]
    assert fake.calls[:3] == ["init", f"create:{nbg_file}", "run"]
    assert fake.last_input.shape == (3, 640, 640)
    assert fake.last_input.dtype == np.uint8


def test_network_is_created_once_across_frames(nbg_file):
    fake = FakeAwnn()
    backend = NPUBackend(ModelConfig(backend="npu", nbg_path=nbg_file), lib=fake)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    backend.detect(frame)
    backend.detect(frame)

    assert fake.calls.count("init") == 1
    assert fake.calls.count("run") == 2


def test_close_releases_the_network_and_driver(nbg_file):
    fake = FakeAwnn()
    backend = NPUBackend(ModelConfig(backend="npu", nbg_path=nbg_file), lib=fake)
    backend.detect(np.zeros((640, 640, 3), dtype=np.uint8))

    backend.close()

    assert fake.calls[-2:] == ["destroy", "uninit"]


def test_missing_nbg_file_gives_a_clear_error(tmp_path):
    config = ModelConfig(backend="npu", nbg_path=str(tmp_path / "missing.nb"))
    backend = NPUBackend(config, lib=FakeAwnn())
    with pytest.raises(FileNotFoundError, match="yolov5.nb"):
        backend.detect(np.zeros((64, 64, 3), dtype=np.uint8))


def test_missing_helper_library_gives_a_clear_error(nbg_file, tmp_path):
    config = ModelConfig(
        backend="npu", nbg_path=nbg_file, awnn_lib_path=str(tmp_path / "libawnn_npu.so")
    )
    backend = NPUBackend(config)
    with pytest.raises(FileNotFoundError, match="build_awnn.sh"):
        backend.detect(np.zeros((64, 64, 3), dtype=np.uint8))
