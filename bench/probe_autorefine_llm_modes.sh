#!/bin/bash
set -e

# Probe AutoRefine LLM call modes on a tiny LawBench subset.
# By default this script restarts AutoRefine vLLM main to avoid hitting a stale
# or wrong service on MAIN_VLLM_PORT.

export BASE_DIR="${BASE_DIR:-/F00120250029/lixiang_share/panghuaiwen_share/legal_R1}"
export CONDA_SH="${CONDA_SH:-/F00120250029/lixiang_share/Data/conda/etc/profile.d/conda.sh}"
export MODEL_PATH="${MODEL_PATH:-${BASE_DIR}/model/AutoRefine-Qwen2.5-7B-Base}"
export MODEL_NAME="${MODEL_NAME:-autorefine-qwen2.5-7b-law-adaptivenote}"
export ADAPTIVE_NOTE_DIR="${ADAPTIVE_NOTE_DIR:-${BASE_DIR}/Adaptive-Note}"

MAIN_VLLM_PORT="${MAIN_VLLM_PORT:-807}"
LIMIT="${LIMIT:-3}"
WORKERS="${WORKERS:-1}"
RESTART_VLLM="${RESTART_VLLM:-1}"
PORT_TIMEOUT="${PORT_TIMEOUT:-1600}"
LAW_INDEX_DIR="${LAW_INDEX_DIR:-${ADAPTIVE_NOTE_DIR}/data/corpus/law}"
LAW_INDEX_PATH="${LAW_INDEX_PATH:-${LAW_INDEX_DIR}/law.index}"
LAW_CHUNK_PATH="${LAW_CHUNK_PATH:-${LAW_INDEX_DIR}/chunk.json}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-BAAI/bge-base-zh-v1.5}"
EMBEDDING_DEVICE="${EMBEDDING_DEVICE:-cpu}"

info() { echo -e "\033[32m[INFO]\033[0m $1"; }
error() { echo -e "\033[31m[ERROR]\033[0m $1"; }

kill_session() {
    if tmux has-session -t "$1" 2>/dev/null; then
        info "清理历史会话: $1"
        tmux kill-session -t "$1"
        sleep 1
    fi
}

wait_for_port() {
    local port=$1
    local timeout=$2
    local session_name=$3
    local start=$(date +%s)
    info "等待端口 $port 就绪（超时 ${timeout}s）..."

    while true; do
        if curl -s -f --noproxy '*' "http://127.0.0.1:$port/health" > /dev/null 2>&1; then
            info "端口 $port 已就绪 (vLLM)"
            return 0
        fi
        if ! tmux has-session -t "$session_name" 2>/dev/null; then
            error "Tmux会话 $session_name 已异常退出。请执行：tmux attach -t $session_name"
            return 1
        fi
        local now=$(date +%s)
        if [ $((now - start)) -ge $timeout ]; then
            error "端口 $port 未在 ${timeout}s 内就绪。请执行：tmux attach -t $session_name"
            return 1
        fi
        sleep 3
    done
}

smoke_test_completions() {
    local code
    code=$(curl -s --noproxy '*' -o /tmp/autorefine_probe_completion_smoke.json -w "%{http_code}" \
        -H 'Content-Type: application/json' \
        -X POST "http://127.0.0.1:${MAIN_VLLM_PORT}/v1/completions" \
        -d "{\"model\":\"${MODEL_NAME}\",\"prompt\":\"你好\",\"max_tokens\":1,\"temperature\":0.0}")
    if [ "$code" != "200" ]; then
        error "/v1/completions smoke test failed with HTTP $code"
        error "Response saved at /tmp/autorefine_probe_completion_smoke.json"
        error "请检查：tmux attach -t autorefine_vllm_main_probe"
        return 1
    fi
    info "/v1/completions smoke test passed."
}

if [ ! -f "$CONDA_SH" ]; then
    error "Conda init script not found: $CONDA_SH"
    exit 1
fi

if [ ! -d "$MODEL_PATH" ]; then
    error "AutoRefine model dir not found: $MODEL_PATH"
    exit 1
fi

if [ ! -f "$LAW_INDEX_PATH" ] || [ ! -f "$LAW_CHUNK_PATH" ]; then
    error "Law FAISS index not found. Run the main script once to build it first."
    exit 1
fi

if [ "$RESTART_VLLM" = "1" ]; then
    kill_session "autorefine_vllm_main"
    kill_session "autorefine_vllm_main_probe"
    tmux new -d -s autorefine_vllm_main_probe "export TRITON_CACHE_DIR=~/.triton/cache_vllm_autorefine_probe; export CUDA_VISIBLE_DEVICES=0; source $CONDA_SH && conda activate vllm_server; export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH; python -m vllm.entrypoints.openai.api_server --model ${MODEL_PATH} --served-model-name ${MODEL_NAME} --port ${MAIN_VLLM_PORT} --gpu-memory-utilization 0.85 --max-model-len 20000 || sleep 86400"
    info "已触发重启 AutoRefine vLLM probe main (Port: ${MAIN_VLLM_PORT}, GPU: 0)"
    wait_for_port "$MAIN_VLLM_PORT" "$PORT_TIMEOUT" "autorefine_vllm_main_probe"
else
    if ! curl -s -f --noproxy '*' "http://127.0.0.1:${MAIN_VLLM_PORT}/health" > /dev/null 2>&1; then
        error "AutoRefine vLLM is not healthy at port ${MAIN_VLLM_PORT}. Set RESTART_VLLM=1 or start vLLM manually."
        exit 1
    fi
fi

smoke_test_completions

source "$CONDA_SH"
conda activate autorefine

# for mode in completion chat qwen_chat_template; do
for mode in qwen_chat_template; do
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
