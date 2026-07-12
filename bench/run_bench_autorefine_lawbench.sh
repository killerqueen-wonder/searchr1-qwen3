#!/bin/bash
set -e

# LawBench-only evaluation for thunlp/Adaptive-Note AutoRefine.
# This script intentionally does not modify or reuse the original model infer
# entrypoint, so other benchmark pipelines remain unaffected.

export BASE_DIR="${BASE_DIR:-/F00120250029/lixiang_share/panghuaiwen_share/legal_R1}"
export CONDA_HOME="${CONDA_HOME:-/data/panghuaiwen/miniconda3}"
export CONDA_SH="${CONDA_SH:-/F00120250029/lixiang_share/Data/conda/etc/profile.d/conda.sh}"

export MODEL_PATH="${MODEL_PATH:-${BASE_DIR}/model/AutoRefine-Qwen2.5-7B-Base}"
export MODEL_NAME="${MODEL_NAME:-autorefine-qwen2.5-7b-law}"
export ADAPTIVE_NOTE_DIR="${ADAPTIVE_NOTE_DIR:-${BASE_DIR}/Adaptive-Note}"

export JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:-/F00120250029/lixiang_share/Models/Qwen3-8B}"
export JUDGE_MODEL_NAME="${JUDGE_MODEL_NAME:-Qwen3-8B-Judge}"
export RERANK_MODEL_PATH="${RERANK_MODEL_PATH:-/F00120250029/lixiang_share/Models/Qwen3-8B}"
export RERANK_MODEL_NAME="${RERANK_MODEL_NAME:-Qwen3-8B}"

WORKERS="${WORKERS:-32}"
AUTOREFINE_MAX_STEP="${AUTOREFINE_MAX_STEP:-3}"
AUTOREFINE_MAX_FAIL_STEP="${AUTOREFINE_MAX_FAIL_STEP:-1}"
AUTOREFINE_TOPK="${AUTOREFINE_TOPK:-5}"
AUTOREFINE_MAX_TOPK="${AUTOREFINE_MAX_TOPK:-15}"

LAW_CORPUS_PATH="${LAW_CORPUS_PATH:-${BASE_DIR}/dataset/dataset/law/法律法规3.0.jsonl}"
# The current unified retriever script requires case_corpus_path at startup.
# AutoRefine only sends legal retrieval requests, so this defaults to the same
# law corpus as a compatibility placeholder instead of loading a case dataset.
DUMMY_CASE_CORPUS_PATH="${DUMMY_CASE_CORPUS_PATH:-${LAW_CORPUS_PATH}}"

RETRIEVE_PATH="http://127.0.0.1:805/retrieve"
RETRIEVER_PORT=805
VLLM_PORT=806
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
        if [ "$port" = "805" ]; then
            if curl -s --noproxy '*' "http://127.0.0.1:$port" > /dev/null 2>&1; then
                info "端口 $port 已就绪 (RAG)"
                return 0
            fi
        else
            if curl -s -f --noproxy '*' "http://127.0.0.1:$port/health" > /dev/null 2>&1; then
                info "端口 $port 已就绪 (vLLM)"
                return 0
            fi
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

tmux_send_commands() {
    local session=$1
    shift
    tmux send-keys -t "$session" "source $CONDA_SH" C-m
    for cmd in "$@"; do
        tmux send-keys -t "$session" "$cmd" C-m
    done
}

if ! command -v tmux &> /dev/null; then
    info "未检测到 tmux，尝试安装..."
    apt update && apt install -y tmux
fi

if ! command -v nvidia-smi &> /dev/null || ! nvidia-smi &> /dev/null; then
    error "致命错误：NVIDIA GPU 未正常挂载或驱动损坏！"
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
info "🚀 AutoRefine LawBench 评测启动"
info "BASE_DIR: ${BASE_DIR}"
info "MODEL_PATH: ${MODEL_PATH}"
info "MODEL_NAME: ${MODEL_NAME}"
info "LAW_CORPUS_PATH: ${LAW_CORPUS_PATH}"
info "========================================================="

kill_session "autorefine_retriever_filter8005"
tmux new-session -d -s autorefine_retriever_filter8005 -n retriever
tmux_send_commands "autorefine_retriever_filter8005" \
    "conda activate retriever_filter" \
    "export TRANSFORMERS_CACHE=${BASE_DIR}/model" \
    "export HF_HUB_CACHE=${BASE_DIR}/model" \
    "export TRANSFORMERS_OFFLINE=1" \
    "export HF_HUB_OFFLINE=1" \
    "cd ${BASE_DIR}/searchr1-qwen3" \
    "bash retrieval_launch_law_text2vec.sh --port $RETRIEVER_PORT --corpus_path '${LAW_CORPUS_PATH}' --case_corpus_path '${DUMMY_CASE_CORPUS_PATH}' --retriever_name hybrid_filter --dictionary_path '${BASE_DIR}/dataset/dataset/dictionary/THUOCL_law.txt' --search_depth 5 --bm25_weight 15 --bm25_weight_factor 2 --bm25_k1 0.15 --bm25_b 0.35 --topk 3 --retriever_model 'shibing624/text2vec-base-chinese-paraphrase' --filter_model ${RERANK_MODEL_PATH} --vllm_url http://127.0.0.1:${VLLM_PORT}/v1/completions --gpu_ids 3 --gpu_memory_limit_per_gpu 8"
info "已触发启动法律 RAG 检索器 (Port: $RETRIEVER_PORT, GPU: 3)"

sleep 15

kill_session "autorefine_vllm_rerank"
tmux new -d -s autorefine_vllm_rerank "export TRITON_CACHE_DIR=~/.triton/cache_vllm_autorefine_rag; export CUDA_VISIBLE_DEVICES=3; source $CONDA_SH && conda activate vllm_server; export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH; python -m vllm.entrypoints.openai.api_server --model ${RERANK_MODEL_PATH} --served-model-name ${RERANK_MODEL_NAME} --port ${VLLM_PORT} --gpu-memory-utilization 0.6 --max-model-len 25000 || sleep 86400"
info "已触发启动 RAG 依赖 vLLM (Port: ${VLLM_PORT}, GPU: 3)"

sleep 15

kill_session "autorefine_vllm_main"
tmux new -d -s autorefine_vllm_main "export TRITON_CACHE_DIR=~/.triton/cache_vllm_autorefine_main; export CUDA_VISIBLE_DEVICES=0; source $CONDA_SH && conda activate vllm_server; export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH; python -m vllm.entrypoints.openai.api_server --model ${MODEL_PATH} --served-model-name ${MODEL_NAME} --port ${MAIN_VLLM_PORT} --gpu-memory-utilization 0.85 --max-model-len 20000 || sleep 86400"
info "已触发启动 AutoRefine vLLM 主路推理 (Port: $MAIN_VLLM_PORT, GPU: 0)"

sleep 15

kill_session "autorefine_vllm_judge"
tmux new -d -s autorefine_vllm_judge "export TRITON_CACHE_DIR=~/.triton/cache_vllm_autorefine_judge; export CUDA_VISIBLE_DEVICES=2; source $CONDA_SH && conda activate vllm_server; export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH; python -m vllm.entrypoints.openai.api_server --model ${JUDGE_MODEL_PATH} --served-model-name ${JUDGE_MODEL_NAME} --port ${JUDGE_VLLM_PORT} --gpu-memory-utilization 0.85 --max-model-len 22000 || sleep 86400"
info "已触发启动 vLLM 裁判模型 (Port: $JUDGE_VLLM_PORT, GPU: 2)"

wait_for_port $VLLM_PORT $PORT_TIMEOUT "autorefine_vllm_rerank"
wait_for_port $RETRIEVER_PORT $PORT_TIMEOUT "autorefine_retriever_filter8005"
wait_for_port $MAIN_VLLM_PORT $PORT_TIMEOUT "autorefine_vllm_main"
wait_for_port $JUDGE_VLLM_PORT $PORT_TIMEOUT "autorefine_vllm_judge"
info "全部服务就绪，开始 AutoRefine LawBench！"

source "$CONDA_SH"
conda activate lawbench

LAWBENCH_PRED_DIR="${BASE_DIR}/lawbench/test/prediction/zero_shot/${MODEL_NAME}"
LAWBENCH_SCORE_DIR="${BASE_DIR}/lawbench/test/result/${MODEL_NAME}_scored"
LAWBENCH_RESULT_PATH="${BASE_DIR}/dataset/result/bench_result/lawbench/${MODEL_NAME}_lawbench.json"

info " -> 1. AutoRefine 推理阶段"
python ${BASE_DIR}/searchr1-qwen3/bench/lawbench/lawbench_autorefine_infer.py \
    --data_dir "${BASE_DIR}/lawbench/test/data/zero_shot" \
    --output_dir "${LAWBENCH_PRED_DIR}" \
    --model_name "${MODEL_NAME}" \
    --vllm_url "http://127.0.0.1:${MAIN_VLLM_PORT}" \
    --retrieve_path "${RETRIEVE_PATH}" \
    --prompt_dir "${ADAPTIVE_NOTE_DIR}/prompts/zh" \
    --max_step ${AUTOREFINE_MAX_STEP} \
    --max_fail_step ${AUTOREFINE_MAX_FAIL_STEP} \
    --topk ${AUTOREFINE_TOPK} \
    --max_top_k ${AUTOREFINE_MAX_TOPK} \
    --workers ${WORKERS}

info " -> 2. 评测阶段"
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
info "🎉 AutoRefine LawBench 评测结束：${LAWBENCH_RESULT_PATH}"
info "========================================================="
