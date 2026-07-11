#!/bin/bash
set -e

# 2xA100-80G version of run_bench_ais_qwen3_AR_without_llm_rerank.sh.
# Main idea:
#   - inference phase: keep the main model on one GPU, share aux services on the other GPU
#   - eval phase: stop inference services, then start judge alone
# This keeps context lengths unchanged and trades wall-clock time/concurrency for memory.

export BASE_DIR="${BASE_DIR:-/F00120250029/lixiang_share/panghuaiwen_share/legal_R1}"
export CONDA_HOME="${CONDA_HOME:-/data/panghuaiwen/miniconda3}"
export CONDA_SH="${CONDA_SH:-/F00120250029/lixiang_share/Data/conda/etc/profile.d/conda.sh}"

export MODEL_PATH="${MODEL_PATH:-/F00120250029/lixiang_share/panghuaiwen_share/legal_R1/model/RL_ckp/legal_exam-ppo-qwen3-8b-RL-7.3-0520-H20/global_step_90/actor/actor_merge}"
export MODEL_NAME="${MODEL_NAME:-qwen3-8b-AR-no_llm_rerank_0711_2gpu}"

export JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:-/F00120250029/lixiang_share/Models/Qwen3-8B}"
export JUDGE_MODEL_NAME="${JUDGE_MODEL_NAME:-Qwen3-8B-Judge}"

export RERANK_MODEL_PATH="${RERANK_MODEL_PATH:-/F00120250029/lixiang_share/Models/Qwen3-8B}"
export RERANK_MODEL_NAME="${RERANK_MODEL_NAME:-Qwen3-8B}"

export SUMMARY_MODEL_PATH="${SUMMARY_MODEL_PATH:-/F00120250029/lixiang_share/Models/Qwen3-8B}"
export SUMMARY_MODEL_NAME="${SUMMARY_MODEL_NAME:-Qwen3-8B}"

WORKERS="${WORKERS:-4}"

GPU_MAIN="${GPU_MAIN:-0}"
GPU_AUX="${GPU_AUX:-1}"

MAIN_GPU_UTIL="${MAIN_GPU_UTIL:-0.85}"
SUMMARY_GPU_UTIL="${SUMMARY_GPU_UTIL:-0.42}"
RAG_VLLM_GPU_UTIL="${RAG_VLLM_GPU_UTIL:-0.28}"
JUDGE_GPU_UTIL="${JUDGE_GPU_UTIL:-0.85}"

MAIN_MAX_NUM_SEQS="${MAIN_MAX_NUM_SEQS:-8}"
SUMMARY_MAX_NUM_SEQS="${SUMMARY_MAX_NUM_SEQS:-4}"
RAG_MAX_NUM_SEQS="${RAG_MAX_NUM_SEQS:-4}"
JUDGE_MAX_NUM_SEQS="${JUDGE_MAX_NUM_SEQS:-8}"
RETRIEVER_GPU_LIMIT="${RETRIEVER_GPU_LIMIT:-8}"

RETRIEVE_PATH="http://127.0.0.1:805/retrieve"
RETRIEVER_PORT=805
VLLM_PORT=806
MAIN_VLLM_PORT=807
SUMMARY_VLLM_PORT=808
JUDGE_VLLM_PORT=809
PORT_TIMEOUT="${PORT_TIMEOUT:-3200}"

info() { echo -e "\033[32m[INFO]\033[0m $1"; }
error() { echo -e "\033[31m[ERROR]\033[0m $1"; }

wait_for_port() {
    local port=$1
    local timeout=$2
    local session_name=$3
    local start
    start=$(date +%s)
    info "Waiting for port ${port} (${timeout}s timeout)..."

    while true; do
        if [ "$port" = "805" ]; then
            if curl -s --noproxy '*' "http://127.0.0.1:$port" > /dev/null 2>&1; then
                info "Port ${port} is ready (RAG)"
                return 0
            fi
        else
            if curl -s -f --noproxy '*' "http://127.0.0.1:$port/health" > /dev/null 2>&1; then
                info "Port ${port} is ready (vLLM)"
                return 0
            fi
        fi

        if ! tmux has-session -t "$session_name" 2>/dev/null; then
            error "Tmux session ${session_name} exited unexpectedly."
            error "Check logs with: tmux attach -t ${session_name}"
            return 1
        fi

        local now
        now=$(date +%s)
        if [ $((now - start)) -ge "$timeout" ]; then
            error "Port ${port} was not ready within ${timeout}s."
            error "Check logs with: tmux attach -t ${session_name}"
            return 1
        fi
        sleep 3
    done
}

kill_session() {
    if tmux has-session -t "$1" 2>/dev/null; then
        info "Stopping tmux session: $1"
        tmux kill-session -t "$1"
        sleep 2
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

stop_inference_services() {
    kill_session "vllm_summary_2gpu"
    kill_session "vllm_main_2gpu"
    kill_session "vllm_rag_2gpu"
    kill_session "retriever_filter8005_2gpu"
}

if ! command -v tmux &> /dev/null; then
    info "tmux is not installed, trying to install it..."
    apt update && apt install -y tmux
fi

conda config --add envs_dirs /data/panghuaiwen/legal_R1/env 2>/dev/null || true

if ! command -v nvidia-smi &> /dev/null || ! nvidia-smi &> /dev/null; then
    error "NVIDIA GPU is not available in this container."
    exit 1
fi

info "========================================================="
info "Starting LawBench on 2xA100-80G"
info "BASE_DIR: ${BASE_DIR}"
info "GPUs: main=${GPU_MAIN}, aux=${GPU_AUX}"
info "Workers: ${WORKERS}"
info "========================================================="

stop_inference_services
kill_session "vllm_judge_2gpu"

tmux new-session -d -s retriever_filter8005_2gpu -n retriever
tmux_send_commands "retriever_filter8005_2gpu" \
    "conda activate retriever_filter" \
    "export TRANSFORMERS_CACHE=${BASE_DIR}/model" \
    "export HF_HUB_CACHE=${BASE_DIR}/model" \
    "export TRANSFORMERS_OFFLINE=1" \
    "export HF_HUB_OFFLINE=1" \
    "cd ${BASE_DIR}/searchr1-qwen3" \
    "bash retrieval_launch_law_text2vec_without_llm_rerank.sh --port $RETRIEVER_PORT --corpus_path '${BASE_DIR}/dataset/dataset/law/法律法规3.0.jsonl' --case_corpus_path '${BASE_DIR}/dataset/dataset/case/lecard_court_psi.jsonl' --retriever_name hybrid_filter --dictionary_path '${BASE_DIR}/dataset/dataset/dictionary/THUOCL_law.txt' --search_depth 5 --bm25_weight 15 --bm25_weight_factor 2 --bm25_k1 0.15 --bm25_b 0.35 --topk 3 --retriever_model 'shibing624/text2vec-base-chinese-paraphrase' --filter_model ${RERANK_MODEL_PATH} --vllm_url http://127.0.0.1:${VLLM_PORT}/v1/completions --gpu_ids ${GPU_AUX} --gpu_memory_limit_per_gpu ${RETRIEVER_GPU_LIMIT}"
info "Started RAG retriever on GPU ${GPU_AUX}, port ${RETRIEVER_PORT}"

sleep 15

tmux new -d -s vllm_rag_2gpu "export TRITON_CACHE_DIR=~/.triton/cache_vllm_rag_2gpu; export CUDA_VISIBLE_DEVICES=${GPU_AUX}; source $CONDA_SH && conda activate vllm_server; export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH; python -m vllm.entrypoints.openai.api_server --model ${RERANK_MODEL_PATH} --served-model-name ${RERANK_MODEL_NAME} --port ${VLLM_PORT} --gpu-memory-utilization ${RAG_VLLM_GPU_UTIL} --max-model-len 25000 --max-num-seqs ${RAG_MAX_NUM_SEQS} || sleep 86400"
info "Started RAG vLLM on GPU ${GPU_AUX}, port ${VLLM_PORT}"

sleep 15

tmux new -d -s vllm_main_2gpu "export TRITON_CACHE_DIR=~/.triton/cache_vllm_main_2gpu; export CUDA_VISIBLE_DEVICES=${GPU_MAIN}; source $CONDA_SH && conda activate vllm_server; export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH; python -m vllm.entrypoints.openai.api_server --model ${MODEL_PATH} --served-model-name ${MODEL_NAME} --port ${MAIN_VLLM_PORT} --gpu-memory-utilization ${MAIN_GPU_UTIL} --max-model-len 20000 --max-num-seqs ${MAIN_MAX_NUM_SEQS} || sleep 86400"
info "Started main vLLM on GPU ${GPU_MAIN}, port ${MAIN_VLLM_PORT}"

sleep 15

tmux new -d -s vllm_summary_2gpu "export TRITON_CACHE_DIR=~/.triton/cache_vllm_summary_2gpu; export CUDA_VISIBLE_DEVICES=${GPU_AUX}; source $CONDA_SH && conda activate vllm_server; export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH; python -m vllm.entrypoints.openai.api_server --model ${SUMMARY_MODEL_PATH} --served-model-name ${SUMMARY_MODEL_NAME} --port ${SUMMARY_VLLM_PORT} --gpu-memory-utilization ${SUMMARY_GPU_UTIL} --max-model-len 20000 --max-num-seqs ${SUMMARY_MAX_NUM_SEQS} || sleep 86400"
info "Started summary vLLM on GPU ${GPU_AUX}, port ${SUMMARY_VLLM_PORT}"

wait_for_port $VLLM_PORT $PORT_TIMEOUT "vllm_rag_2gpu"
wait_for_port $RETRIEVER_PORT $PORT_TIMEOUT "retriever_filter8005_2gpu"
wait_for_port $MAIN_VLLM_PORT $PORT_TIMEOUT "vllm_main_2gpu"
wait_for_port $SUMMARY_VLLM_PORT $PORT_TIMEOUT "vllm_summary_2gpu"

info "================== LawBench inference =================="
source $CONDA_SH
conda activate lawbench

LAWBENCH_PRED_DIR="${BASE_DIR}/lawbench/test/prediction/zero_shot/${MODEL_NAME}"
LAWBENCH_SCORE_DIR="${BASE_DIR}/lawbench/test/result/${MODEL_NAME}_scored"
LAWBENCH_RESULT_PATH="${BASE_DIR}/dataset/result/bench_result/lawbench/${MODEL_NAME}_lawbench.json"

python ${BASE_DIR}/searchr1-qwen3/bench/lawbench/lawbench_infer.py \
    --data_dir "${BASE_DIR}/lawbench/test/data/zero_shot" \
    --output_dir "${LAWBENCH_PRED_DIR}" \
    --model_name "${MODEL_NAME}" \
    --vllm_url "http://127.0.0.1:${MAIN_VLLM_PORT}" \
    --summary_port ${SUMMARY_VLLM_PORT} \
    --retrieve_path "${RETRIEVE_PATH}" \
    --max_turn 12 --topk 10 --workers ${WORKERS} --retriever

info "Inference finished. Releasing inference services before eval..."
stop_inference_services
sleep 10

info "================== LawBench eval =================="
tmux new -d -s vllm_judge_2gpu "export TRITON_CACHE_DIR=~/.triton/cache_vllm_judge_2gpu; export CUDA_VISIBLE_DEVICES=${GPU_MAIN}; source $CONDA_SH && conda activate vllm_server; export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH; python -m vllm.entrypoints.openai.api_server --model ${JUDGE_MODEL_PATH} --served-model-name ${JUDGE_MODEL_NAME} --port ${JUDGE_VLLM_PORT} --gpu-memory-utilization ${JUDGE_GPU_UTIL} --max-model-len 22000 --max-num-seqs ${JUDGE_MAX_NUM_SEQS} || sleep 86400"
info "Started judge vLLM on GPU ${GPU_MAIN}, port ${JUDGE_VLLM_PORT}"
wait_for_port $JUDGE_VLLM_PORT $PORT_TIMEOUT "vllm_judge_2gpu"

python ${BASE_DIR}/searchr1-qwen3/bench/lawbench/lawbench_eval.py \
    --input_dir "${LAWBENCH_PRED_DIR}" \
    --output_dir "${LAWBENCH_SCORE_DIR}" \
    --judge_port ${JUDGE_VLLM_PORT} \
    --judge_model_name "${JUDGE_MODEL_NAME}" \
    --workers ${WORKERS}

python ${BASE_DIR}/searchr1-qwen3/bench/lawbench/lawbench_result.py \
    --score_dir "${LAWBENCH_SCORE_DIR}" \
    --output_path "${LAWBENCH_RESULT_PATH}"

info "========================================================="
info "LawBench finished."
info "Result: ${LAWBENCH_RESULT_PATH}"
info "Judge session is still running: tmux attach -t vllm_judge_2gpu"
info "========================================================="
