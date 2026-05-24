#!/bin/bash
set -e

# ================= 1. 命令行参数解析 (新增 API 配置) =================
JUDGE_API_KEY=""
JUDGE_API_URL="https://api.shubiaobiao.cn/v1/chat/completions" # 根据你的实际报错日志设为默认值
JUDGE_MODEL_NAME="deepseek-chat" # 替换为你实际使用的模型名

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --api_key) 
            JUDGE_API_KEY="$2"; shift ;;
        --api_url) 
            JUDGE_API_URL="$2"; shift ;;
        --judge_model) 
            JUDGE_MODEL_NAME="$2"; shift ;;
        *) 
            echo "未知的参数: $1"
            echo "用法: bash run_bench_ais_qwen3_AR_hybrid.sh --api_key <Token> [--api_url <URL>] [--judge_model <模型名>]"
            exit 1 ;;
    esac
    shift
done

if [ -z "$JUDGE_API_KEY" ]; then
    echo -e "\033[31m[ERROR] 缺少必要的参数 --api_key！\033[0m"
    exit 1
fi

# ================= 2. 配置全局变量 =================
export BASE_DIR="${BASE_DIR:-/F00120250029/lixiang_share/panghuaiwen_share/legal_R1}"
export CONDA_HOME="${CONDA_HOME:-/data/panghuaiwen/miniconda3}"

# ----------------- 模型路径与名称配置 -----------------
# 1. 主路推理模型 (对应 MAIN_VLLM_PORT)
export MODEL_PATH="/F00120250029/lixiang_share/panghuaiwen_share/legal_R1/model/RL_ckp/legal_exam-ppo-qwen3-8b-RL-7.3-0520-H20/global_step_90/actor/actor_merge"
export MODEL_NAME="qwen3-8b-AR-0521-hybrid-0524"

# 2. Rerank 模型配置 (对应 VLLM_PORT)
export RERANK_MODEL_PATH="${RERANK_MODEL_PATH:-/F00120250029/lixiang_share/Models/Qwen3-8B}"
export RERANK_MODEL_NAME="${RERANK_MODEL_NAME:-Qwen3-8B}"

# 3. 总结模型配置 (对应 SUMMARY_VLLM_PORT)
export SUMMARY_MODEL_PATH="${SUMMARY_MODEL_PATH:-/F00120250029/lixiang_share/Models/Qwen3-8B}"
export SUMMARY_MODEL_NAME="${SUMMARY_MODEL_NAME:-Qwen3-8B}"

WORKERS="${WORKERS:-32}"

# 接口与并发配置 (去除了 JUDGE_VLLM_PORT)
RETRIEVE_PATH="http://127.0.0.1:805/retrieve" 
RETRIEVER_PORT=805
VLLM_PORT=806
MAIN_VLLM_PORT=807
SUMMARY_VLLM_PORT=808

# 等待超时设置（秒）
PORT_TIMEOUT=1600  

# ================= 3. 环境初始化 =================
export CONDA_SH="/F00120250029/lixiang_share/Data/conda/etc/profile.d/conda.sh"

info() { echo -e "\033[32m[INFO]\033[0m $1"; }
error() { echo -e "\033[31m[ERROR]\033[0m $1"; }

# ----------------- 基础设施函数 -----------------
wait_for_port() {
    local port=$1
    local timeout=$2
    local session_name=$3
    local start=$(date +%s)
    info "等待端口 $port 就绪（超时 ${timeout}s）..."
    
    while true; do
        if [ "$port" = "$RETRIEVER_PORT" ]; then
            if curl -s --noproxy '*' "http://127.0.0.1:$port/" > /dev/null 2>&1; then
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
            error "Tmux会话 $session_name 已异常退出！启动失败。"
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

# ----------------- 环境预检 -----------------
if ! command -v tmux &> /dev/null; then
    apt update && apt install -y tmux
fi
conda config --add envs_dirs /data/panghuaiwen/legal_R1/env 2>/dev/null || true

# ================= 4. 启动后台服务 =================
info "========================================================="
info " 🚀 全量 Bench 自动化评测启动 (Hybrid RAG + API Judge)"
info " 工作目录: ${BASE_DIR}"
info " 裁判模型: ${JUDGE_MODEL_NAME} (API: ${JUDGE_API_URL})"
info "========================================================="

# --- 4.1 启动 RAG 检索器及依赖 vLLM (GPU 3) ---
kill_session "retriever_filter8005"
tmux new-session -d -s retriever_filter8005 -n retriever
tmux_send_commands "retriever_filter8005" \
    "conda activate retriever_filter" \
    "export TRANSFORMERS_CACHE=${BASE_DIR}/model" \
    "export HF_HUB_CACHE=${BASE_DIR}/model" \
    "export TRANSFORMERS_OFFLINE=1" \
    "export HF_HUB_OFFLINE=1" \
    "cd ${BASE_DIR}/searchr1-qwen3" \
    "bash retrieval_launch_law_text2vec.sh --port $RETRIEVER_PORT --corpus_path '${BASE_DIR}/dataset/dataset/law/法律法规3.0.jsonl' --case_corpus_path '${BASE_DIR}/dataset/dataset/case/lecard_court_psi.jsonl' --retriever_name hybrid --dictionary_path '${BASE_DIR}/dataset/dataset/dictionary/THUOCL_law.txt' --search_depth 5 --bm25_weight 15 --bm25_weight_factor 2 --bm25_k1 0.15 --bm25_b 0.35 --topk 3 --retriever_model 'shibing624/text2vec-base-chinese-paraphrase' --filter_model ${RERANK_MODEL_PATH} --vllm_url http://127.0.0.1:${VLLM_PORT}/v1/completions --gpu_ids 3 --gpu_memory_limit_per_gpu 8"
info "已触发启动 RAG 检索器 (Port: $RETRIEVER_PORT, GPU: 3)"
sleep 15

kill_session "vllm"
tmux new -d -s vllm "export TRITON_CACHE_DIR=~/.triton/cache_vllm_rag; export CUDA_VISIBLE_DEVICES=3; source $CONDA_SH && conda activate vllm_server; export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH; python -m vllm.entrypoints.openai.api_server --model ${RERANK_MODEL_PATH} --served-model-name ${RERANK_MODEL_NAME} --port ${VLLM_PORT} --gpu-memory-utilization 0.9 --max-model-len 10000 || sleep 86400"
info "已触发启动 RAG 依赖的 vLLM (Port: ${VLLM_PORT}, GPU: 3)"
sleep 15

# --- 4.2 启动主路推理、总结 vLLM (GPU 0, 1) ---
kill_session "vllm_main"
tmux new -d -s vllm_main "export TRITON_CACHE_DIR=~/.triton/cache_vllm_main; export CUDA_VISIBLE_DEVICES=0; source $CONDA_SH && conda activate vllm_server; export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH; python -m vllm.entrypoints.openai.api_server --model ${MODEL_PATH} --served-model-name ${MODEL_NAME} --port ${MAIN_VLLM_PORT} --gpu-memory-utilization 0.9 --max-model-len 20000 || sleep 86400"
info "已触发启动 vLLM 主路推理 (Port: $MAIN_VLLM_PORT, GPU: 0)"
sleep 15

kill_session "vllm_summary"
tmux new -d -s vllm_summary "export TRITON_CACHE_DIR=~/.triton/cache_vllm_summary; export CUDA_VISIBLE_DEVICES=1; source $CONDA_SH && conda activate vllm_server; export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH; python -m vllm.entrypoints.openai.api_server --model ${SUMMARY_MODEL_PATH} --served-model-name ${SUMMARY_MODEL_NAME} --port ${SUMMARY_VLLM_PORT} --gpu-memory-utilization 0.9 --max-model-len 20000 || sleep 86400"
info "已触发启动 vLLM 总结模型 (Port: $SUMMARY_VLLM_PORT, GPU: 1)"

# 【移除】这里删除了裁判模型 vllm_judge 的拉起逻辑，释放了 GPU 2 资源

info "等待所有服务拉起..."
wait_for_port $VLLM_PORT $PORT_TIMEOUT "vllm"
wait_for_port $RETRIEVER_PORT $PORT_TIMEOUT "retriever_filter8005"
wait_for_port $MAIN_VLLM_PORT $PORT_TIMEOUT "vllm_main"
wait_for_port $SUMMARY_VLLM_PORT $PORT_TIMEOUT "vllm_summary"
info "全部服务就绪，开始执行基准测试！🚀"

# ================= 5. UCL Bench =================
info "================== [1/3] UCL Bench =================="

source $CONDA_SH
conda activate searchr1_new

UCL_RES_PATH="${BASE_DIR}/dataset/result/res_result/${MODEL_NAME}_ucl_eval_result.json"
# 【修改】Score 和 Result 路径加上了 api_judge_0524 后缀
UCL_SCORE_PATH="${BASE_DIR}/dataset/result/score_result/${MODEL_NAME}_ucl_score_api_judge_0524.json"
UCL_RESULT_PATH="${BASE_DIR}/dataset/result/bench_result/UCL/${MODEL_NAME}_ucl_api_judge_0524.json"
UCL_CHATGPT_REF="${BASE_DIR}/dataset/result/res_result/qwen3_8B_eval_result.json" 

info " -> 1. 推理阶段"
python ${BASE_DIR}/searchr1-qwen3/bench/ucl/ucl_infer.py \
    --data_path "${BASE_DIR}/UCL-bench/dataset/legal_data_sample.json" \
    --result_path "${UCL_RES_PATH}" \
    --model_name "${MODEL_NAME}" \
    --vllm_url "http://127.0.0.1:${MAIN_VLLM_PORT}" \
    --summary_port ${SUMMARY_VLLM_PORT} \
    --retrieve_path "${RETRIEVE_PATH}" \
    --max_turn 12 --topk 10 --workers ${WORKERS} --retriever True

info " -> 2. 评测阶段 (API Judge)"
# 【修改】使用 api judge 脚本并传入 api 凭证
python ${BASE_DIR}/searchr1-qwen3/bench/ucl/ucl_eval_api_judge.py \
    --chatgpt_result_path "${UCL_CHATGPT_REF}" \
    --model_result_path "${UCL_RES_PATH}" \
    --datasource_path "${BASE_DIR}/UCL-bench/dataset/legal_data_sample.json" \
    --result_path "${UCL_SCORE_PATH}" \
    --api_key "${JUDGE_API_KEY}" \
    --api_url "${JUDGE_API_URL}" \
    --judge_model_name "${JUDGE_MODEL_NAME}" \
    --workers ${WORKERS}

info " -> 3. 统计汇总"
python ${BASE_DIR}/searchr1-qwen3/bench/ucl/ucl_result.py \
    --score_path "${UCL_SCORE_PATH}" \
    --inference_path "${UCL_RES_PATH}" \
    --output_path "${UCL_RESULT_PATH}"

# ================= 6. LawBench =================
info "================== [2/3] LawBench =================="

info " -> 切换为 LawBench 环境..."
conda deactivate
conda activate lawbench

LAWBENCH_PRED_DIR="${BASE_DIR}/lawbench/test/prediction/zero_shot/${MODEL_NAME}"
# 【修改】增加后缀
LAWBENCH_SCORE_DIR="${BASE_DIR}/lawbench/test/result/${MODEL_NAME}_scored_api_judge_0524"
LAWBENCH_RESULT_PATH="${BASE_DIR}/dataset/result/bench_result/lawbench/${MODEL_NAME}_lawbench_api_judge_0524.json"

info " -> 1. 推理阶段"
python ${BASE_DIR}/searchr1-qwen3/bench/lawbench/lawbench_infer.py \
    --data_dir "${BASE_DIR}/lawbench/test/data/zero_shot" \
    --output_dir "${LAWBENCH_PRED_DIR}" \
    --model_name "${MODEL_NAME}" \
    --vllm_url "http://127.0.0.1:${MAIN_VLLM_PORT}" \
    --summary_port ${SUMMARY_VLLM_PORT} \
    --retrieve_path "${RETRIEVE_PATH}" \
    --max_turn 12 --topk 10 --workers ${WORKERS} --retriever

info " -> 2. 评测阶段 (API Judge)"
# 【修改】替换为 api judge 脚本
python ${BASE_DIR}/searchr1-qwen3/bench/lawbench/lawbench_eval_api_judge.py \
    --input_dir "${LAWBENCH_PRED_DIR}" \
    --output_dir "${LAWBENCH_SCORE_DIR}" \
    --api_key "${JUDGE_API_KEY}" \
    --api_url "${JUDGE_API_URL}" \
    --judge_model_name "${JUDGE_MODEL_NAME}" \
    --workers ${WORKERS}

info " -> 3. 统计汇总"
python ${BASE_DIR}/searchr1-qwen3/bench/lawbench/lawbench_result.py \
    --score_dir "${LAWBENCH_SCORE_DIR}" \
    --output_path "${LAWBENCH_RESULT_PATH}"

# ================= 7. LexEval =================
info "================== [3/3] LexEval =================="

info " -> 切换回 searchr1_new 环境..."
conda deactivate
conda activate searchr1_new

LEXEVAL_PRED_DIR="${BASE_DIR}/LexEval/model_output/zero_shot/${MODEL_NAME}"
# 【修改】增加后缀
LEXEVAL_SCORE_DIR="${BASE_DIR}/LexEval/evaluation_output/${MODEL_NAME}_scored_api_judge_0524"
LEXEVAL_RESULT_PATH="${BASE_DIR}/dataset/result/bench_result/lexeval/${MODEL_NAME}_lexeval_api_judge_0524.json"

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

info " -> 2. 评测阶段 (API Judge)"
# 【修改】替换为 api judge 脚本
python ${BASE_DIR}/searchr1-qwen3/bench/lexeval/lexeval_eval_api_judge.py \
    --input_dir "${LEXEVAL_PRED_DIR}" \
    --output_dir "${LEXEVAL_SCORE_DIR}" \
    --api_key "${JUDGE_API_KEY}" \
    --api_url "${JUDGE_API_URL}" \
    --judge_model_name "${JUDGE_MODEL_NAME}" \
    --workers ${WORKERS}

info " -> 3. 统计汇总"
python ${BASE_DIR}/searchr1-qwen3/bench/lexeval/lexeval_result.py \
    --score_dir "${LEXEVAL_SCORE_DIR}" \
    --output_path "${LEXEVAL_RESULT_PATH}"

# ================= 8. 收尾清理 =================
info "========================================================="
info "🎉 所有评测任务圆满结束！各项报告已生成于："
info "   - UCL:      ${UCL_RESULT_PATH}"
info "   - LawBench: ${LAWBENCH_RESULT_PATH}"
info "   - LexEval:  ${LEXEVAL_RESULT_PATH}"
info "========================================================="