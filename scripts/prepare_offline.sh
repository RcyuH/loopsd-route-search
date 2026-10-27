#!/usr/bin/env bash
# Run ONLINE on Linux matching the target's architecture, Python and OS baseline.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
PYTHON=${PYTHON:-python3.13}
BUNDLE=${BUNDLE:-"$ROOT/offline-bundle"}
PREP_VENV=${PREP_VENV:-"$ROOT/.venv-prepare"}
LENS_REV=4d045c03186b90afdc815db46b671f6a21bc7e98

if [[ $(uname -s) != Linux ]]; then
  echo "Prepare this bundle on Linux matching the target server, not macOS." >&2
  exit 1
fi
if [[ -e "$BUNDLE" || -e "$PREP_VENV" ]]; then
  echo "Choose fresh BUNDLE and PREP_VENV paths; refusing to mix resolutions." >&2
  exit 1
fi
"$PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 13), "Use Python 3.13 on both machines"'
NEMO_REV=$(git -C "$ROOT/../nemo-rl" rev-parse HEAD)
mkdir -p "$BUNDLE/wheelhouse"
BUNDLE=$(cd "$BUNDLE" && pwd)
"$PYTHON" -m venv "$PREP_VENV"
PREP_VENV=$(cd "$PREP_VENV" && pwd)
PYPREP="$PREP_VENV/bin/python"

"$PYPREP" -m pip download --only-binary=:all: --dest "$BUNDLE/wheelhouse" pip
"$PYPREP" -m pip install --no-index --find-links "$BUNDLE/wheelhouse" --upgrade pip
# Explicit CUDA index prevents accidentally bundling a CPU/CUDA-12.6 torch build.
"$PYPREP" -m pip download --only-binary=:all: --dest "$BUNDLE/wheelhouse" \
  --index-url https://download.pytorch.org/whl/cu132 'torch==2.13.0+cu132'
# pip ignores [tool.uv.sources], so build the exact Lens source explicitly.
"$PYPREP" -m pip wheel --no-deps --wheel-dir "$BUNDLE/wheelhouse" \
  "nemo-lens @ git+https://github.com/NVIDIA-NeMo/Lens.git@$LENS_REV"
"$PYPREP" -m pip wheel --wheel-dir "$BUNDLE/wheelhouse" \
  --find-links "$BUNDLE/wheelhouse" -r requirements.txt
# Install exclusively from the bundle to resolve and record all transitives.
"$PYPREP" -m pip install --no-index --find-links "$BUNDLE/wheelhouse" -r requirements.txt
"$PYPREP" -m pip freeze > "$BUNDLE/requirements.lock.txt"
"$PYPREP" - "$BUNDLE" "$LENS_REV" "$NEMO_REV" <<'PY'
import importlib.metadata as m
import json
import pathlib
import platform
import sys

bundle = pathlib.Path(sys.argv[1])
metadata = {
    "python": sys.version,
    "platform": platform.platform(),
    "machine": platform.machine(),
    "libc": platform.libc_ver(),
    "torch": m.version("torch"),
    "lens_revision": sys.argv[2],
    "nemo_revision": sys.argv[3],
}
(bundle / "environment.json").write_text(json.dumps(metadata, indent=2) + "\n")
PY
(cd "$BUNDLE" && sha256sum wheelhouse/*.whl requirements.lock.txt environment.json > SHA256SUMS)
echo "Prepared $BUNDLE; transfer it with the code, NeMo checkout, model and data."
