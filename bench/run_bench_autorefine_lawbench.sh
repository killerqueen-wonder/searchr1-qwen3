#!/bin/bash
set -e

# LawBench-only evaluation for thunlp/Adaptive-Note AutoRefine.
# This version uses Adaptive-Note's original Chinese CRUD-style RAG backend:
# BGE embeddings + local FAISS index. search-R1 hybrid retrieval and LLM rerank
# are intentionally not used in the AutoRefine inference stage.

export BASE_DIR="${BASE_DIR:-/F00120250029/lixiang_share/panghuaiwen_share/legal_R1}"
export CONDA_HOME="${CONDA_HOME:-/data/panghuaiwen/miniconda3}"
export CONDA_SH="${CONDA_SH:-/F00120250029/lixiang_share/Data/conda/etc/profile.d/conda.sh}"

export MODEL_PATH="${MODEL_PATH:-${BASE_DIR}/model/AutoRefine-Qwen2.5-7B-Base}"
# export MODEL_NAME="${MODEL_NAME:-autorefine-qwen2.5-7b-law-adaptivenote}"
export MODEL_NAME="${MODEL_NAME:-autorefine-qwen2.5-7b-law-4-turn}"
export ADAPTIVE_NOTE_DIR="${ADAPTIVE_NOTE_DIR:-${BASE_DIR}/Adaptive-Note}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-${BASE_DIR}/model}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${BASE_DIR}/model}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${BASE_DIR}/model}"

export JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:-/F00120250029/lixiang_share/Models/Qwen3-8B}"
export JUDGE_MODEL_NAME="${JUDGE_MODEL_NAME:-Qwen3-8B-Judge}"

WORKERS="${WORKERS:-32}"
AUTOREFINE_MAX_STEP="${AUTOREFINE_MAX_STEP:-5}"
AUTOREFINE_MAX_FAIL_STEP="${AUTOREFINE_MAX_FAIL_STEP:-3}"
AUTOREFINE_TOPK="${AUTOREFINE_TOPK:-5}"
AUTOREFINE_MAX_TOPK="${AUTOREFINE_MAX_TOPK:-15}"

LAW_CORPUS_PATH="${LAW_CORPUS_PATH:-${BASE_DIR}/dataset/dataset/law/法律法规3.0.jsonl}"
LAW_INDEX_DIR="${LAW_INDEX_DIR:-${ADAPTIVE_NOTE_DIR}/data/corpus/law}"
LAW_INDEX_PATH="${LAW_INDEX_PATH:-${LAW_INDEX_DIR}/law.index}"
LAW_CHUNK_PATH="${LAW_CHUNK_PATH:-${LAW_INDEX_DIR}/chunk.json}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-BAAI/bge-base-zh-v1.5}"
EMBEDDING_CUDA_VISIBLE_DEVICES="${EMBEDDING_CUDA_VISIBLE_DEVICES:-3}"
EMBEDDING_DEVICE="${EMBEDDING_DEVICE:-cuda:0}"

MAIN_VLLM_PORT=807
JUDGE_VLLM_PORT=809
PORT_TIMEOUT="${PORT_TIMEOUT:-1600}"

info() { echo -e "\033[32m[INFO]\033[0m $1"; }
error() { echo -e "\033[31m[ERROR]\033[0m $1"; }

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
            error "Tmux会话 $session_name 已异常退出！"
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

kill_session() {
    if tmux has-session -t "$1" 2>/dev/null; then
        info "清理历史会话: $1"
        tmux kill-session -t "$1"
        sleep 1
    fi
}

if ! command -v tmux &> /dev/null; then
    info "未检测到 tmux，尝试安装..."
    apt update && apt install -y tmux
fi

if ! command -v nvidia-smi &> /dev/null || ! nvidia-smi &> /dev/null; then
    error "致命错误：NVIDIA GPU 未正常挂载或驱动损坏！"
    exit 1
fi

if [ ! -f "$CONDA_SH" ]; then
    error "Conda init script not found: $CONDA_SH"
    exit 1
fi

if [ ! -d "$ADAPTIVE_NOTE_DIR" ]; then
    error "Adaptive-Note 仓库不存在: $ADAPTIVE_NOTE_DIR"
    error "请先运行: bash ${BASE_DIR}/searchr1-qwen3/bench/install_autorefine.sh"
    exit 1
fi

if [ ! -d "$MODEL_PATH" ]; then
    error "AutoRefine 模型目录不存在: $MODEL_PATH"
    error "请先运行: bash ${BASE_DIR}/searchr1-qwen3/bench/install_autorefine.sh"
    exit 1
fi

info "========================================================="
info "🚀 Adaptive-Note AutoRefine LawBench 评测启动"
info "BASE_DIR: ${BASE_DIR}"
info "MODEL_PATH: ${MODEL_PATH}"
info "MODEL_NAME: ${MODEL_NAME}"
info "LAW_CORPUS_PATH: ${LAW_CORPUS_PATH}"
info "LAW_INDEX_PATH: ${LAW_INDEX_PATH}"
info "LAW_CHUNK_PATH: ${LAW_CHUNK_PATH}"
info "RAG backend: Adaptive-Note BGE + FAISS (no search-R1, no LLM rerank)"
info "========================================================="

source "$CONDA_SH"

if [ ! -f "$LAW_INDEX_PATH" ] || [ ! -f "$LAW_CHUNK_PATH" ]; then
    info "未发现 Adaptive-Note 法律 FAISS 索引，开始构建..."
    conda activate autorefine
    mkdir -p "$LAW_INDEX_DIR"
    CUDA_VISIBLE_DEVICES="$EMBEDDING_CUDA_VISIBLE_DEVICES" python ${BASE_DIR}/searchr1-qwen3/bench/lawbench/build_autorefine_law_index.py \
        --corpus_path "$LAW_CORPUS_PATH" \
        --output_dir "$LAW_INDEX_DIR" \
        --embedding_model "$EMBEDDING_MODEL" \
        --device "$EMBEDDING_DEVICE"
else
    info "复用已有 Adaptive-Note 法律 FAISS 索引。"
fi

kill_session "autorefine_vllm_main"
tmux new -d -s autorefine_vllm_main "export TRITON_CACHE_DIR=~/.triton/cache_vllm_autorefine_main; export CUDA_VISIBLE_DEVICES=0; source $CONDA_SH && conda activate vllm_server; export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH; python -m vllm.entrypoints.openai.api_server --model ${MODEL_PATH} --served-model-name ${MODEL_NAME} --port ${MAIN_VLLM_PORT} --gpu-memory-utilization 0.85 --max-model-len 20000 || sleep 86400"
info "已触发启动 AutoRefine vLLM 主路推理 (Port: $MAIN_VLLM_PORT, GPU: 0)"

sleep 15

kill_session "autorefine_vllm_judge"
tmux new -d -s autorefine_vllm_judge "export TRITON_CACHE_DIR=~/.triton/cache_vllm_autorefine_judge; export CUDA_VISIBLE_DEVICES=2; source $CONDA_SH && conda activate vllm_server; export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH; python -m vllm.entrypoints.openai.api_server --model ${JUDGE_MODEL_PATH} --served-model-name ${JUDGE_MODEL_NAME} --port ${JUDGE_VLLM_PORT} --gpu-memory-utilization 0.85 --max-model-len 22000 || sleep 86400"
info "已触发启动 vLLM 裁判模型 (Port: $JUDGE_VLLM_PORT, GPU: 2)"

wait_for_port $MAIN_VLLM_PORT $PORT_TIMEOUT "autorefine_vllm_main"
wait_for_port $JUDGE_VLLM_PORT $PORT_TIMEOUT "autorefine_vllm_judge"
info "全部服务就绪，开始 Adaptive-Note AutoRefine LawBench！"

conda activate autorefine

LAWBENCH_PRED_DIR="${BASE_DIR}/lawbench/test/prediction/zero_shot/${MODEL_NAME}"
LAWBENCH_SCORE_DIR="${BASE_DIR}/lawbench/test/result/${MODEL_NAME}_scored"
LAWBENCH_RESULT_PATH="${BASE_DIR}/dataset/result/bench_result/lawbench/${MODEL_NAME}_lawbench.json"

info " -> 1. Adaptive-Note AutoRefine 推理阶段"
CUDA_VISIBLE_DEVICES="$EMBEDDING_CUDA_VISIBLE_DEVICES" python ${BASE_DIR}/searchr1-qwen3/bench/lawbench/lawbench_autorefine_infer.py \
    --data_dir "${BASE_DIR}/lawbench/test/data/zero_shot" \
    --output_dir "${LAWBENCH_PRED_DIR}" \
    --model_name "${MODEL_NAME}" \
    --vllm_url "http://127.0.0.1:${MAIN_VLLM_PORT}" \
    --prompt_dir "${ADAPTIVE_NOTE_DIR}/prompts/zh" \
    --law_index_path "${LAW_INDEX_PATH}" \
    --law_chunk_path "${LAW_CHUNK_PATH}" \
    --embedding_model "${EMBEDDING_MODEL}" \
    --embedding_device "${EMBEDDING_DEVICE}" \
    --max_step ${AUTOREFINE_MAX_STEP} \
    --max_fail_step ${AUTOREFINE_MAX_FAIL_STEP} \
    --topk ${AUTOREFINE_TOPK} \
    --max_top_k ${AUTOREFINE_MAX_TOPK} \
    --workers ${WORKERS}

info " -> 2. 评测阶段"
conda activate lawbench
python ${BASE_DIR}/searchr1-qwen3/bench/lawbench/lawbench_eval.py \
    --input_dir "${LAWBENCH_PRED_DIR}" \
    --output_dir "${LAWBENCH_SCORE_DIR}" \
    --judge_port ${JUDGE_VLLM_PORT} \
    --judge_model_name "${JUDGE_MODEL_NAME}" \
    --workers ${WORKERS}

info " -> 3. 统计汇总"
python ${BASE_DIR}/searchr1-qwen3/bench/lawbench/lawbench_result.py \
    --score_dir "${LAWBENCH_SCORE_DIR}" \
    --output_path "${LAWBENCH_RESULT_PATH}"

info "========================================================="
info "🎉 Adaptive-Note AutoRefine LawBench 评测结束：${LAWBENCH_RESULT_PATH}"
info "========================================================="
