#!/bin/bash
set -euo pipefail

# Install Adaptive-Note and download AutoRefine-Qwen2.5-7B-Base.
# This script is designed for the remote Linux server, not for local Windows.

export BASE_DIR="${BASE_DIR:-/F00120250029/lixiang_share/panghuaiwen_share/legal_R1}"
export CONDA_HOME="${CONDA_HOME:-/data/panghuaiwen/miniconda3}"
export CONDA_SH="${CONDA_SH:-${CONDA_HOME}/etc/profile.d/conda.sh}"

export ADAPTIVE_NOTE_DIR="${ADAPTIVE_NOTE_DIR:-${BASE_DIR}/Adaptive-Note}"
export AUTOREFINE_MODEL_REPO="${AUTOREFINE_MODEL_REPO:-yrshi/AutoRefine-Qwen2.5-7B-Base}"
export AUTOREFINE_MODEL_DIR="${AUTOREFINE_MODEL_DIR:-${BASE_DIR}/model/AutoRefine-Qwen2.5-7B-Base}"
export AUTOREFINE_ENV_NAME="${AUTOREFINE_ENV_NAME:-autorefine}"

# China-friendly defaults. Override these before running if your server has a
# different mirror policy.
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-${BASE_DIR}/model}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${BASE_DIR}/model}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${BASE_DIR}/model}"
export PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"

info() { echo -e "\033[32m[INFO]\033[0m $1"; }
error() { echo -e "\033[31m[ERROR]\033[0m $1"; }

if [ ! -f "$CONDA_SH" ]; then
    error "Conda init script not found: $CONDA_SH"
    exit 1
fi

source "$CONDA_SH"

mkdir -p "${BASE_DIR}/model"

if [ ! -d "$ADAPTIVE_NOTE_DIR/.git" ]; then
    info "Cloning Adaptive-Note into ${ADAPTIVE_NOTE_DIR}"
    git clone https://github.com/thunlp/Adaptive-Note.git "$ADAPTIVE_NOTE_DIR"
else
    info "Adaptive-Note already exists: ${ADAPTIVE_NOTE_DIR}"
fi

if ! conda env list | awk '{print $1}' | grep -qx "$AUTOREFINE_ENV_NAME"; then
    info "Creating conda env: ${AUTOREFINE_ENV_NAME}"
    conda create -n "$AUTOREFINE_ENV_NAME" python=3.10 -y
fi

conda activate "$AUTOREFINE_ENV_NAME"
python -m pip install --upgrade pip
python -m pip install -r "${ADAPTIVE_NOTE_DIR}/requirements.txt"
python -m pip install huggingface_hub requests tqdm

info "Downloading ${AUTOREFINE_MODEL_REPO} to ${AUTOREFINE_MODEL_DIR}"
huggingface-cli download \
    "$AUTOREFINE_MODEL_REPO" \
    --local-dir "$AUTOREFINE_MODEL_DIR" \
    --local-dir-use-symlinks False \
    --resume-download

info "Done."
info "Adaptive-Note repo: ${ADAPTIVE_NOTE_DIR}"
info "AutoRefine model: ${AUTOREFINE_MODEL_DIR}"

