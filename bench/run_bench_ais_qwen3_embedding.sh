#!/bin/bash
set -e

# ================= 1. 配置全局变量 =================
export BASE_DIR="${BASE_DIR:-/F00120250029/lixiang_share/panghuaiwen_share/legal_R1}"
export CONDA_HOME="${CONDA_HOME:-/data/panghuaiwen/miniconda3}"

# ----------------- 模型路径与名称配置 -----------------
# 1. Embedding 模型配置 (RAG 组件)
export EMBEDDING_MODEL_PATH="/F00120250029/lixiang_share/panghuaiwen_share/legal_R1/model/models--Qwen--Qwen3-Embedding-8B"
# 注意：MODEL_NAME 必须包含 "embedding"，以触发 shared_agent 的专属线路
export MODEL_NAME="qwen3-embedding-0522-1557"

# 2. 裁判模型配置 (对应 JUDGE_VLLM_PORT)
export JUDGE_MODEL_PATH="/F00120250029/lixiang_share/Models/Qwen3-8B"
export JUDGE_MODEL_NAME="Qwen3-8B-Judge"

# 3. 总结模型配置 (对应 SUMMARY_VLLM_PORT)
export SUMMARY_MODEL_PATH="/F00120250029/lixiang_share/Models/Qwen3-8B"
export SUMMARY_MODEL_NAME="Qwen3-8B"

WORKERS="${WORKERS:-32}"

# 接口与并发配置
RETRIEVE_PATH="http://127.0.0.1:805/retrieve" 
RETRIEVER_PORT=805
SUMMARY_VLLM_PORT=808
JUDGE_VLLM_PORT=809

PORT_TIMEOUT=1600  

# ====== 核心环境变量：开启直连模式 & 开启专属 Prompt ======
export USE_DIRECT_API="true"
export EMBEDDING_PIPELINE="true"

# ================= 2. 环境初始化 =================
export CONDA_SH="/F00120250029/lixiang_share/Data/conda/etc/profile.d/conda.sh"

info() { echo -e "\033[32m[INFO]\033[0m $1"; }
error() { echo -e "\033[31m[ERROR]\033[0m $1"; }

wait_for_port() {
    local port=$1
    local timeout=$2
    local session_name=$3
    local start=$(date +%s)
    info "等待端口 $port 就绪（超时 ${timeout}s）..."
    
    while true; do
        if [ "$port" = "$RETRIEVER_PORT" ]; then
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
            error "端口 $port 未在 ${timeout}s 内就绪。"
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

# ================= 3. 启动后台服务 =================
info "========================================================="
info " 🚀 Embedding 专线 Bench 自动化评测启动"
info " 工作目录: ${BASE_DIR}"
info "========================================================="

# --- 3.1 启动纯 Qwen3-Embedding RAG 检索器 (GPU 3) ---
kill_session "retriever_emb_805"
tmux new-session -d -s retriever_emb_805 -n retriever

tmux_send_commands "retriever_emb_805" \
    "conda activate retriever_filter" \
    "export TRANSFORMERS_OFFLINE=1" \
    "export HF_HUB_OFFLINE=1" \
    "export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True" \
    "cd ${BASE_DIR}/searchr1-qwen3" \
    "python search_r1/search/retrieval_server_qwen3_embedding.py --corpus_path '${BASE_DIR}/dataset/dataset/law/法律法规3.0.jsonl' --retriever_model '${EMBEDDING_MODEL_PATH}' --batch_size 4 --port $RETRIEVER_PORT --gpu_ids 3 --max_concurrency 4"
info "已触发启动 Qwen3-Embedding 检索器 (Port: $RETRIEVER_PORT, GPU: 3)"

sleep 15

# --- 3.2 启动总结与评测 vLLM (独占 1, 2 卡) ---
kill_session "vllm_summary_emb"
tmux new -d -s vllm_summary_emb "export TRITON_CACHE_DIR=~/.triton/cache_vllm_summary; export CUDA_VISIBLE_DEVICES=1; source $CONDA_SH && conda activate vllm_server; export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH; python -m vllm.entrypoints.openai.api_server --model ${SUMMARY_MODEL_PATH} --served-model-name ${SUMMARY_MODEL_NAME} --port ${SUMMARY_VLLM_PORT} --gpu-memory-utilization 0.85 --max-model-len 20000 || sleep 86400"
info "已触发启动 vLLM 总结模型 (Port: $SUMMARY_VLLM_PORT, GPU: 1)"

sleep 15

kill_session "vllm_judge_emb"
tmux new -d -s vllm_judge_emb "export TRITON_CACHE_DIR=~/.triton/cache_vllm_judge; export CUDA_VISIBLE_DEVICES=2; source $CONDA_SH && conda activate vllm_server; export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH; python -m vllm.entrypoints.openai.api_server --model ${JUDGE_MODEL_PATH} --served-model-name ${JUDGE_MODEL_NAME} --port ${JUDGE_VLLM_PORT} --gpu-memory-utilization 0.85 --max-model-len 22000 || sleep 86400"
info "已触发启动 vLLM 裁判模型 (Port: $JUDGE_VLLM_PORT, GPU: 2)"

info "等待所有服务拉起..."
wait_for_port $RETRIEVER_PORT $PORT_TIMEOUT "retriever_emb_805"
wait_for_port $SUMMARY_VLLM_PORT $PORT_TIMEOUT "vllm_summary_emb"
wait_for_port $JUDGE_VLLM_PORT $PORT_TIMEOUT "vllm_judge_emb"
info "全部服务就绪，开始执行基准测试！🚀"


# ================= 4. UCL Bench =================
info "================== [1/3] UCL Bench =================="

source $CONDA_SH
conda activate searchr1_new

UCL_RES_PATH="${BASE_DIR}/dataset/result/res_result/${MODEL_NAME}_ucl_eval_result.json"
UCL_SCORE_PATH="${BASE_DIR}/dataset/result/score_result/${MODEL_NAME}_ucl_score.json"
UCL_RESULT_PATH="${BASE_DIR}/dataset/result/bench_result/UCL/${MODEL_NAME}_ucl.json"
UCL_CHATGPT_REF="${BASE_DIR}/dataset/result/res_result/qwen3_8B_eval_result.json" 

info " -> 1. 推理阶段 (Embedding + Summary)"
python ${BASE_DIR}/searchr1-qwen3/bench/ucl/ucl_infer.py \
    --data_path "${BASE_DIR}/UCL-bench/dataset/legal_data_sample.json" \
    --result_path "${UCL_RES_PATH}" \
    --model_name "${MODEL_NAME}" \
    --vllm_url "http://127.0.0.1:${SUMMARY_VLLM_PORT}" \
    --summary_port ${SUMMARY_VLLM_PORT} \
    --retrieve_path "${RETRIEVE_PATH}" \
    --max_turn 1 --topk 10 --workers ${WORKERS} --retriever True

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

info " -> 1. 推理阶段 (Embedding + Summary)"
python ${BASE_DIR}/searchr1-qwen3/bench/lawbench/lawbench_infer.py \
    --data_dir "${BASE_DIR}/lawbench/test/data/zero_shot" \
    --output_dir "${LAWBENCH_PRED_DIR}" \
    --model_name "${MODEL_NAME}" \
    --vllm_url "http://127.0.0.1:${SUMMARY_VLLM_PORT}" \
    --summary_port ${SUMMARY_VLLM_PORT} \
    --retrieve_path "${RETRIEVE_PATH}" \
    --max_turn 1 --topk 10 --workers ${WORKERS} --retriever

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

info " -> 1. 推理阶段 (Embedding + Summary)"
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
                --vllm_url "http://127.0.0.1:${SUMMARY_VLLM_PORT}" \
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
info "🎉 Embedding 专线评测任务圆满结束！各项报告已生成于："
info "   - UCL:      ${UCL_RESULT_PATH}"
info "   - LawBench: ${LAWBENCH_RESULT_PATH}"
info "   - LexEval:  ${LEXEVAL_RESULT_PATH}"
info "========================================================="