#!/usr/bin/env python3
"""Fetch the pretrained MobileNet-SSD (VOC0712) weights used by the CPU backend.

Weights aren't committed to the repo (23MB binary, not something to carry
in git history). This pulls the same files chuanqi305/MobileNet-SSD
publishes, verified reachable at the time of writing this script.
"""
from __future__ import annotations

import pathlib
import sys
import urllib.request

BASE_URL = "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master"
FILES = {
    "deploy.prototxt": f"{BASE_URL}/deploy.prototxt",
    "mobilenet_iter_73000.caffemodel": f"{BASE_URL}/mobilenet_iter_73000.caffemodel",
}


def main() -> int:
    models_dir = pathlib.Path(__file__).resolve().parent.parent / "models"
    models_dir.mkdir(exist_ok=True)

    for filename, url in FILES.items():
        dest = models_dir / filename
        if dest.exists() and dest.stat().st_size > 0:
            print(f"skip (already present): {dest}")
            continue
        print(f"downloading {url} -> {dest}")
        urllib.request.urlretrieve(url, dest)
        print(f"  ok, {dest.stat().st_size} bytes")

    return 0


if __name__ == "__main__":
    sys.exit(main())
