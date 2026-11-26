#!/usr/bin/env bash
# Run on the EPFL Linux server. Does not select a GPU or download model weights.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${METONYMY_VENV:-$HOME/venvs/metonymy-group2}"
PYTHON_BIN="${METONYMY_PYTHON:-python3}"

"$PYTHON_BIN" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Use Python 3.12 to match Group 1"'
if [ ! -x "$ENV_DIR/bin/python" ]; then
    if "$PYTHON_BIN" -c 'import ensurepip' 2>/dev/null; then
        "$PYTHON_BIN" -m venv "$ENV_DIR"
    elif [ -x /opt/anaconda3/bin/python ]; then
        "$PYTHON_BIN" -m venv --without-pip "$ENV_DIR"
        /opt/anaconda3/bin/python -m pip --python "$ENV_DIR/bin/python" install pip
    else
        echo 'Ask the administrator for python3.12-venv (ensurepip is unavailable).'
        exit 1
    fi
fi
"$ENV_DIR/bin/python" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Existing environment is not Python 3.12"'
"$ENV_DIR/bin/python" -m pip install --upgrade pip
"$ENV_DIR/bin/python" -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu126
"$ENV_DIR/bin/python" -m pip install -e "$PROJECT_ROOT[encoder]"
"$ENV_DIR/bin/python" -m pip check
CUDA_VISIBLE_DEVICES="" "$ENV_DIR/bin/python" -c 'import torch; assert torch.version.cuda == "12.6", torch.__version__; print("PyTorch:", torch.__version__, "CUDA runtime:", torch.version.cuda)'
cd "$PROJECT_ROOT"
CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    "$ENV_DIR/bin/python" -m pytest -q
printf '\nActivate with: source "%s/bin/activate"\n' "$ENV_DIR"
echo 'Environment ready. No GPU job or model download was started.'
