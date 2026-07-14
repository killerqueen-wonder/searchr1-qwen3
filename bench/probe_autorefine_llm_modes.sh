#!/bin/bash
set -e

# Probe AutoRefine LLM call modes on a tiny LawBench subset.
# It assumes AutoRefine vLLM is already running on MAIN_VLLM_PORT.

export BASE_DIR="${BASE_DIR:-/F00120250029/lixiang_share/panghuaiwen_share/legal_R1}"
export CONDA_SH="${CONDA_SH:-/F00120250029/lixiang_share/Data/conda/etc/profile.d/conda.sh}"
export MODEL_NAME="${MODEL_NAME:-autorefine-qwen2.5-7b-law-adaptivenote}"
export ADAPTIVE_NOTE_DIR="${ADAPTIVE_NOTE_DIR:-${BASE_DIR}/Adaptive-Note}"

MAIN_VLLM_PORT="${MAIN_VLLM_PORT:-807}"
LIMIT="${LIMIT:-5}"
WORKERS="${WORKERS:-1}"
LAW_INDEX_DIR="${LAW_INDEX_DIR:-${ADAPTIVE_NOTE_DIR}/data/corpus/law}"
LAW_INDEX_PATH="${LAW_INDEX_PATH:-${LAW_INDEX_DIR}/law.index}"
LAW_CHUNK_PATH="${LAW_CHUNK_PATH:-${LAW_INDEX_DIR}/chunk.json}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-BAAI/bge-base-zh-v1.5}"
EMBEDDING_DEVICE="${EMBEDDING_DEVICE:-cpu}"

info() { echo -e "\033[32m[INFO]\033[0m $1"; }
error() { echo -e "\033[31m[ERROR]\033[0m $1"; }

if [ ! -f "$CONDA_SH" ]; then
    error "Conda init script not found: $CONDA_SH"
    exit 1
fi

if ! curl -s -f --noproxy '*' "http://127.0.0.1:${MAIN_VLLM_PORT}/health" > /dev/null 2>&1; then
    error "AutoRefine vLLM is not healthy at port ${MAIN_VLLM_PORT}. Start run_bench_autorefine_lawbench.sh first, or start vLLM manually."
    exit 1
fi

if [ ! -f "$LAW_INDEX_PATH" ] || [ ! -f "$LAW_CHUNK_PATH" ]; then
    error "Law FAISS index not found. Run the main script once to build it first."
    exit 1
fi

source "$CONDA_SH"
conda activate autorefine

for mode in completion chat qwen_chat_template; do
    OUT_DIR="${BASE_DIR}/lawbench/test/prediction/zero_shot/${MODEL_NAME}_probe_${mode}"
    rm -rf "$OUT_DIR"
    info "Running probe mode=${mode}, limit=${LIMIT}, workers=${WORKERS}"
    python ${BASE_DIR}/searchr1-qwen3/bench/lawbench/lawbench_autorefine_infer.py \
        --data_dir "${BASE_DIR}/lawbench/test/data/zero_shot" \
        --output_dir "$OUT_DIR" \
        --model_name "${MODEL_NAME}" \
        --vllm_url "http://127.0.0.1:${MAIN_VLLM_PORT}" \
        --prompt_dir "${ADAPTIVE_NOTE_DIR}/prompts/zh" \
        --law_index_path "${LAW_INDEX_PATH}" \
        --law_chunk_path "${LAW_CHUNK_PATH}" \
        --embedding_model "${EMBEDDING_MODEL}" \
        --embedding_device "${EMBEDDING_DEVICE}" \
        --llm_api_mode "$mode" \
        --limit "$LIMIT" \
        --workers "$WORKERS"
    info "Saved probe output: $OUT_DIR"
done

info "Probe finished. Compare thinking.llm_api_mode, note_log[0].note, raw_final_answer, and prediction across *_probe_* dirs."
