#!/usr/bin/env bash
# Local metric weather env (snow/rain/fog/sandstorm). Optional if you use
# --weather-gpu on Modal instead. Removal does not need this.
set -euo pipefail

PIPE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$PIPE/.." && pwd)"
PI3="$ROOT/third_party/Pi3"
AERIALMETRIC="$ROOT/third_party/AerialMetric"
CHECKPOINT="${AERIALMETRIC_CHECKPOINT:-$ROOT/checkpoints/Moge2-Aerial.pt}"
if [[ -n "${UV_BIN:-}" ]]; then
  UV="$UV_BIN"
elif command -v uv >/dev/null 2>&1; then
  UV="$(command -v uv)"
elif [[ -x "/workspace/.tooling/uv/uv" ]]; then
  UV="/workspace/.tooling/uv/uv"
else
  echo "uv not found. Install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi

if [[ ! -f "$PI3/pyproject.toml" || ! -f "$AERIALMETRIC/MoGe/pyproject.toml" ]]; then
  echo "Missing submodules. Run: git submodule update --init third_party/Pi3 third_party/AerialMetric"
  exit 1
fi

"$UV" venv --python 3.11 "$ROOT/.venv"
"$UV" pip install --python "$ROOT/.venv/bin/python" \
  torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cu128
"$UV" pip install --python "$ROOT/.venv/bin/python" \
  "imageio[ffmpeg]" huggingface_hub numpy opencv-python-headless pillow safetensors scipy warp-lang
"$UV" pip install --python "$ROOT/.venv/bin/python" --no-deps -e "$PI3"

"$UV" venv --python 3.11 "$AERIALMETRIC/.venv"
"$UV" pip install --python "$AERIALMETRIC/.venv/bin/python" \
  torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cu128
"$UV" pip install --python "$AERIALMETRIC/.venv/bin/python" -e "$AERIALMETRIC/MoGe"

if [[ "${1:-}" == "--download-checkpoint" ]]; then
  mkdir -p "$(dirname "$CHECKPOINT")"
  curl -L --fail --retry 3 \
    -o "$CHECKPOINT.partial" \
    "https://huggingface.co/datasets/Kuiee/AerialMetric-ECCV2026/resolve/main/weights/Moge2-Aerial.pt"
  mv "$CHECKPOINT.partial" "$CHECKPOINT"
fi

echo "Weather env ready."
echo "  $ROOT/.venv/bin/python $PIPE/router.py --help"
