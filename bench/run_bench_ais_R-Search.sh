#!/bin/bash
set -e

# ================= 1. 配置全局变量 =================
export BASE_DIR="${BASE_DIR:-/F00120250029/lixiang_share/panghuaiwen_share/legal_R1}"
export CONDA_HOME="${CONDA_HOME:-/data/panghuaiwen/miniconda3}"

# 修改点 1: 更新为新模型的 Name 和 Path
export MODEL_NAME="${MODEL_NAME:-R-Search-0515}"
export MODEL_PATH="/F00120250029/lixiang_share/panghuaiwen_share/legal_R1/model/R-Search"
export JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:-/F00120250029/lixiang_share/Models/Qwen3-8B}"
export JUDGE_MODEL_NAME="${JUDGE_MODEL_NAME:-Qwen3-8B-Judge}"

WORKERS="${WORKERS:-16}"

# 端口配置
MAIN_VLLM_PORT=8017
JUDGE_VLLM_PORT=8019
SUMMARY_VLLM_PORT=8008 # 保留虚假端口供 argparse 使用
PORT=81
PORT_TIMEOUT=1600  

# 开启强制直通模式 (极其重要：将屏蔽 RAG 与 Summary)
export USE_DIRECT_API="true"
export RETRIEVE_PATH="http://127.0.0.1:81/retrieve"
export CONDA_SH="/F00120250029/lixiang_share/Data/conda/etc/profile.d/conda.sh"

info() { echo -e "\033[32m[INFO]\033[0m $1"; }
error() { echo -e "\033[31m[ERROR]\033[0m $1"; }

tmux_send_commands() {
    local session=$1
    shift
    tmux send-keys -t "$session" "source $CONDA_SH" C-m
    for cmd in "$@"; do
        tmux send-keys -t "$session" "$cmd" C-m
    done
}

wait_for_port() {
    local port=$1
    local timeout=$2
    local session_name=$3
    local start=$(date +%s)
    info "等待端口 $port 就绪（超时 ${timeout}s）..."
    
    while true; do
        # 兼容你的 FastAPI /health 和 vLLM 的 /health
        if curl -s --noproxy '*' "http://127.0.0.1:$port/health" > /dev/null 2>&1; then
            info "端口 $port 已就绪"
            return 0
        fi
        
        local now=$(date +%s)
        if [ $((now - start)) -ge $timeout ]; then
            error "端口 $port 未在 ${timeout}s 内就绪。"
            return 1
        fi
        sleep 3
    done
}

kill_session() {
    if tmux has-session -t "$1" 2>/dev/null; then
        tmux kill-session -t "$1"
        sleep 1
    fi
}

# ----------------- 环境预检 -----------------
if ! command -v tmux &> /dev/null; then
    info "未检测到 tmux，尝试安装..."

    apt update && apt install -y tmux
fi

conda config --add envs_dirs /data/panghuaiwen/legal_R1/env 2>/dev/null || true

if ! command -v nvidia-smi &> /dev/null || ! nvidia-smi &> /dev/null; then
    error "致命错误：NVIDIA GPU 未正常挂载或驱动损坏！容器环境异常！"
    exit 1
fi

info "========================================================="
info " 🚀 R-Search 专属评测启动"
info "========================================================="

# --- 2. 启动服务 ---
# 启动主路推理 (GPU 2)
kill_session "r_vllm_main"
tmux new -d -s r_vllm_main "export TRITON_CACHE_DIR=~/.triton/cache_vllm_main; export CUDA_VISIBLE_DEVICES=2; source $CONDA_SH && conda activate vllm_server; export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH; python -m vllm.entrypoints.openai.api_server --model ${MODEL_PATH} --served-model-name ${MODEL_NAME} --port ${MAIN_VLLM_PORT} --gpu-memory-utilization 0.85 --max-model-len 10096 --trust-remote-code --dtype bfloat16 || sleep 86400"
info "已触发启动 vLLM 主路推理 (Port: $MAIN_VLLM_PORT)"


kill_session "r_retriever_81"
tmux new-session -d -s r_retriever_81 -n retriever
tmux_send_commands "r_retriever_81" \
    "conda activate retriever_filter" \
    "export TRANSFORMERS_CACHE=/F00120250029/lixiang_share/panghuaiwen_share/legal_R1/model" \
    "export HF_HUB_CACHE=/F00120250029/lixiang_share/panghuaiwen_share/legal_R1/model" \
    "export TRANSFORMERS_OFFLINE=1" \
    "export HF_HUB_OFFLINE=1" \
    "cd /F00120250029/lixiang_share/panghuaiwen_share/legal_R1/searchr1-qwen3" \
    "git pull origin main" \
    "python search_r1/search/retrieval_server_A2S.py \
        --index_path '/F00120250029/lixiang_share/panghuaiwen_share/legal_R1/dataset/dataset/case/e5_Flat.index' \
        --corpus_path '/F00120250029/lixiang_share/panghuaiwen_share/legal_R1/dataset/dataset/law/法律法规3.0.jsonl' \
        --topk 5 \
        --port $PORT \
        --gpu_ids 3 --gpu_memory_limit_per_gpu 5" \
        

# 启动裁判模型 (GPU 3)
kill_session "r_vllm_judge"
tmux new -d -s r_vllm_judge "export CUDA_VISIBLE_DEVICES=3; \
    source $CONDA_SH && conda activate vllm_server; \
    python -m vllm.entrypoints.openai.api_server \
    --model ${JUDGE_MODEL_PATH} \
    --served-model-name ${JUDGE_MODEL_NAME} \
    --port ${JUDGE_VLLM_PORT} \
    --gpu-memory-utilization 0.9 \
    --max-model-len 20000 --enforce-eager || sleep 86400"
info "已触发启动 Judge 裁判服务 "

wait_for_port $MAIN_VLLM_PORT $PORT_TIMEOUT "r_vllm_main"
wait_for_port $JUDGE_VLLM_PORT $PORT_TIMEOUT "r_vllm_judge"
wait_for_port $PORT $PORT_TIMEOUT "r_retriever_81"
info "所有服务就绪，开始执行基准测试！🚀"

# --- 3. 运行评测 ---
info "================== [1/3] UCL Bench =================="
source $CONDA_SH
conda activate searchr1_new

UCL_RES_PATH="${BASE_DIR}/dataset/result/res_result/${MODEL_NAME}_ucl_eval_result.json"
UCL_SCORE_PATH="${BASE_DIR}/dataset/result/score_result/${MODEL_NAME}_ucl_score.json"
UCL_RESULT_PATH="${BASE_DIR}/dataset/result/bench_result/UCL/${MODEL_NAME}_ucl.json"
UCL_CHATGPT_REF="${BASE_DIR}/dataset/result/res_result/qwen3_8B_eval_result.json" 

info " -> 1. 推理阶段"
python ${BASE_DIR}/searchr1-qwen3/bench/ucl/ucl_infer.py \
    --data_path "${BASE_DIR}/UCL-bench/dataset/legal_data_sample.json" \
    --result_path "${UCL_RES_PATH}" \
    --model_name "${MODEL_NAME}" \
    --vllm_url "http://127.0.0.1:${MAIN_VLLM_PORT}" \
    --summary_port ${SUMMARY_VLLM_PORT} \
    --retrieve_path "${RETRIEVE_PATH}" \
    --retriever true \
    --max_turn 10 --workers ${WORKERS}

info " -> 2. 评测阶段"
python ${BASE_DIR}/searchr1-qwen3/bench/ucl/ucl_eval.py \
    --chatgpt_result_path "${UCL_CHATGPT_REF}" \
    --model_result_path "${UCL_RES_PATH}" \
    --datasource_path "${BASE_DIR}/UCL-bench/dataset/legal_data_sample.json" \
    --result_path "${UCL_SCORE_PATH}" \
    --judge_port ${JUDGE_VLLM_PORT} \
    --judge_model_name "${JUDGE_MODEL_NAME}" \
    --workers ${WORKERS}

info " -> 3. 统计汇总"
python ${BASE_DIR}/searchr1-qwen3/bench/ucl/ucl_result.py \
    --score_path "${UCL_SCORE_PATH}" \
    --inference_path "${UCL_RES_PATH}" \
    --output_path "${UCL_RESULT_PATH}"

# ================= 5. LawBench =================
info "================== [2/3] LawBench =================="

info " -> 切换为 LawBench 环境..."
source $CONDA_SH
conda activate lawbench

LAWBENCH_PRED_DIR="${BASE_DIR}/lawbench/test/prediction/zero_shot/${MODEL_NAME}"
LAWBENCH_SCORE_DIR="${BASE_DIR}/lawbench/test/result/${MODEL_NAME}_scored"
LAWBENCH_RESULT_PATH="${BASE_DIR}/dataset/result/bench_result/lawbench/${MODEL_NAME}_lawbench.json"

info " -> 1. 推理阶段"
python ${BASE_DIR}/searchr1-qwen3/bench/lawbench/lawbench_infer.py \
    --data_dir "${BASE_DIR}/lawbench/test/data/zero_shot" \
    --output_dir "${LAWBENCH_PRED_DIR}" \
    --model_name "${MODEL_NAME}" \
    --vllm_url "http://127.0.0.1:${MAIN_VLLM_PORT}" \
    --summary_port ${SUMMARY_VLLM_PORT} \
    --retrieve_path "${RETRIEVE_PATH}" \
    --max_turn 12 --topk 10 --workers ${WORKERS} --retriever

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

# ================= 6. LexEval =================
info "================== [3/3] LexEval =================="

info " -> 切换回 searchr1_new 环境..."
source $CONDA_SH
conda activate searchr1_new

LEXEVAL_PRED_DIR="${BASE_DIR}/LexEval/model_output/zero_shot/${MODEL_NAME}"
LEXEVAL_SCORE_DIR="${BASE_DIR}/LexEval/evaluation_output/${MODEL_NAME}_scored"
LEXEVAL_RESULT_PATH="${BASE_DIR}/dataset/result/bench_result/lexeval/${MODEL_NAME}_lexeval.json"

info " -> 1. 推理阶段"

for folder_id in {1..6}; do
    if [ "$folder_id" -eq 1 ]; then max_file=3
    elif [ "$folder_id" -eq 2 ]; then max_file=5
    elif [ "$folder_id" -eq 3 ]; then max_file=6
    elif [ "$folder_id" -eq 4 ]; then max_file=2
    elif [ "$folder_id" -eq 5 ]; then max_file=4
    elif [ "$folder_id" -eq 6 ]; then max_file=3
    fi
    for (( j=1; j<=$max_file; j++ )); do
        FILE_PATH="${BASE_DIR}/LexEval/data/${folder_id}_${j}.json"
        if [ -f "$FILE_PATH" ]; then
            python ${BASE_DIR}/searchr1-qwen3/bench/lexeval/lexeval_infer.py \
                --f_path "$FILE_PATH" \
                --model_name "${MODEL_NAME}" \
                --output_dir "${LEXEVAL_PRED_DIR}" \
                --vllm_url "http://127.0.0.1:${MAIN_VLLM_PORT}" \
                --summary_port ${SUMMARY_VLLM_PORT} \
                --retrieve_path "${RETRIEVE_PATH}" \
                --workers ${WORKERS}
        fi
    done
done

info " -> 2. 评测阶段"
python ${BASE_DIR}/searchr1-qwen3/bench/lexeval/lexeval_eval.py \
    --input_dir "${LEXEVAL_PRED_DIR}" \
    --output_dir "${LEXEVAL_SCORE_DIR}" \
    --judge_port ${JUDGE_VLLM_PORT} \
    --judge_model_name "${JUDGE_MODEL_NAME}" \
    --workers ${WORKERS}

info " -> 3. 统计汇总"
python ${BASE_DIR}/searchr1-qwen3/bench/lexeval/lexeval_result.py \
    --score_dir "${LEXEVAL_SCORE_DIR}" \
    --output_path "${LEXEVAL_RESULT_PATH}"

# ================= 7. 收尾清理 =================
info "========================================================="
info "🎉 所有评测任务圆满结束！各项报告已生成于："
info "   - UCL:      ${UCL_RESULT_PATH}"
info "   - LawBench: ${LAWBENCH_RESULT_PATH}"
info "   - LexEval:  ${LEXEVAL_RESULT_PATH}"
info "========================================================="