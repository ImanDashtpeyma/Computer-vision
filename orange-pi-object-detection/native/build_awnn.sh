#!/usr/bin/env bash
# Build libawnn_npu.so from Allwinner's ai-sdk. Run this ON the Orange Pi (aarch64).
#
#   native/build_awnn.sh /path/to/ai-sdk
#
# The vendor sources (awnn_lib.c, VIPLite headers/libraries) are compiled from your
# own copy of the SDK and are not part of this repository.
set -euo pipefail

SDK="${1:-${AI_SDK_DIR:-}}"
if [[ -z "$SDK" || ! -d "$SDK/examples/libawnn_viplite" ]]; then
  echo "usage: $0 /path/to/ai-sdk   (or set AI_SDK_DIR)" >&2
  exit 1
fi

# A733 reports NPU_VERSION=v3 / NPU_SW_VERSION=v2.0 in ai-sdk/machinfo/a733/config.mk
VIP_LIB="$SDK/viplite-tina/lib/aarch64-none-linux-gnu/v2.0"
SRC="$SDK/examples/libawnn_viplite"
OUT="$(cd "$(dirname "$0")" && pwd)/libawnn_npu.so"

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "warning: this is $(uname -m), not aarch64 -- linking the vendor libraries will fail off-board." >&2
fi

gcc -shared -fPIC -O2 -o "$OUT" \
  "$SRC/awnn_lib.c" "$SRC/awnn_quantize.c" \
  -I"$VIP_LIB/inc" -I"$SRC" \
  -L"$VIP_LIB" -lNBGlinker -lVIPhal -lpthread -lm \
  -Wl,-rpath,"$VIP_LIB"

echo "built $OUT"
