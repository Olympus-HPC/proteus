#!/usr/bin/env bash
set -euo pipefail

WHEEL="${1:?wheel path is required}"
DEST_DIR="${2:?destination directory is required}"
PLAT="${AUDITWHEEL_PLAT:-manylinux_2_28_x86_64}"
REPORT_PATH="${DEST_DIR}/$(basename "${WHEEL%.whl}").auditwheel.txt"
PLAT_FLOOR="${PLAT#manylinux_}"
PLAT_FLOOR="${PLAT_FLOOR%_x86_64}"

mkdir -p "${DEST_DIR}"

SHOW_OUTPUT="$(auditwheel show "${WHEEL}")"
printf '%s\n' "${SHOW_OUTPUT}" | tee "${REPORT_PATH}"

python3 - <<'PY' "${PLAT_FLOOR}" "${REPORT_PATH}"
import re
import sys
from pathlib import Path

target = tuple(int(part) for part in sys.argv[1].split("_"))
report = Path(sys.argv[2]).read_text()

matches = re.findall(r"manylinux_(\d+)_(\d+)_x86_64", report)
if not matches:
    print("Wheel is not compatible with any manylinux_x86_64 platform", file=sys.stderr)
    raise SystemExit(1)

required = max((int(major), int(minor)) for major, minor in matches)
if required > target:
    print(
        f"Wheel requires newer manylinux floor {required[0]}_{required[1]} than {target[0]}_{target[1]}",
        file=sys.stderr,
    )
    raise SystemExit(1)
PY

auditwheel repair --plat "${PLAT}" -w "${DEST_DIR}" "${WHEEL}"
