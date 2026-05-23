#!/bin/bash
set -e

# ================= 1. 命令行参数解析 =================
# 设置 Judge API 的默认值
JUDGE_API_KEY=""
JUDGE_API_URL="https://api.deepseek.com"
JUDGE_MODEL_NAME="deepseek-chat"

# 解析传入的参数
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
            echo "用法: bash run_bench_ais_legalone_api_judge.sh --api_key <Token> [--api_url <URL>] [--judge_model <模型名>]"
            exit 1 ;;
    esac
    shift
done

# 校验必填项
if [ -z "$JUDGE_API_KEY" ]; then
    echo -e "\033[31m[ERROR] 缺少必要的参数 --api_key！\033[0m"
    echo "用法: bash run_bench_ais_legalone_api_judge.sh --api_key <Token> [--api_url <URL>] [--judge_model <模型名>]"
    exit 1
fi

# ================= 2. 配置全局变量 =================
export BASE_DIR="${BASE_DIR:-/F00120250029/lixiang_share/panghuaiwen_share/legal_R1}"
export CONDA_HOME="${CONDA_HOME:-/data/panghuaiwen/miniconda3}"

export MODEL_NAME="${MODEL_NAME:-legalone-0517}"
export MODEL_PATH="/F00120250029/lixiang_share/panghuaiwen_share/legal_R1/model/models--CSHaitao--LegalOne-8B"

WORKERS="${WORKERS:-16}"

# 端口配置
MAIN_VLLM_PORT=8007
SUMMARY_VLLM_PORT=8008 # 保留虚假端口供 argparse 使用
PORT_TIMEOUT=1600  

# 开启强制直通模式 (极其重要：将屏蔽 RAG 与 Summary)
export USE_DIRECT_API="true"

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
        if curl -s -f --noproxy '*' "http://127.0.0.1:$port/health" > /dev/null 2>&1; then
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
info " 🚀 legalone 极速直通评测启动 (无需 RAG 与 Summary)"
info " 方案：GPU 1 [本地推理] | API [裁判: ${JUDGE_MODEL_NAME}]"
info " API 地址: ${JUDGE_API_URL}"
info "========================================================="

# --- 3. 启动服务 ---
# 启动主路推理 (本地大模型)
# kill_session "vllm_main"
# tmux new -d -s vllm_main "export TRITON_CACHE_DIR=~/.triton/cache_vllm_main; export CUDA_VISIBLE_DEVICES=1; source $CONDA_SH && conda activate vllm_server; export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\$LD_LIBRARY_PATH; python -m vllm.entrypoints.openai.api_server --model ${MODEL_PATH} --served-model-name ${MODEL_NAME} --port ${MAIN_VLLM_PORT} --gpu-memory-utilization 0.9 --max-model-len 10000 || sleep 86400"
# info "已触发启动 vLLM 主路推理 (Port: $MAIN_VLLM_PORT, GPU: 1)"

# wait_for_port $MAIN_VLLM_PORT $PORT_TIMEOUT "vllm_main"
# info "本地主推理服务就绪，开始执行基准测试！🚀"

# --- 4. 运行评测 ---
info "================== [1/3] UCL Bench =================="
source $CONDA_SH
conda activate searchr1_new

UCL_RES_PATH="${BASE_DIR}/dataset/result/res_result/${MODEL_NAME}_ucl_eval_result.json"
UCL_SCORE_PATH="${BASE_DIR}/dataset/result/score_result/${MODEL_NAME}_ucl_score_api_judge.json"
UCL_RESULT_PATH="${BASE_DIR}/dataset/result/bench_result/UCL/${MODEL_NAME}_ucl_api_judge.json"
UCL_CHATGPT_REF="/F00120250029/lixiang_share/panghuaiwen_share/legal_R1/dataset/result/res_result/deepseek_eval_result.json" 

info " -> 1. 推理阶段"
# python ${BASE_DIR}/searchr1-qwen3/bench/ucl/ucl_infer.py \
#     --data_path "${BASE_DIR}/UCL-bench/dataset/legal_data_sample.json" \
#     --result_path "${UCL_RES_PATH}" \
#     --model_name "${MODEL_NAME}" 
#     --vllm_url "http://127.0.0.1:${MAIN_VLLM_PORT}" \
#     --summary_port ${SUMMARY_VLLM_PORT} \
#     --retrieve_path "" \
#     --max_turn 1 --workers ${WORKERS}

info " -> 2. 评测阶段 (API 并发打分)"
python ${BASE_DIR}/searchr1-qwen3/bench/ucl/ucl_new_std/ucl_eval_api_judge.py \
    --chatgpt_result_path "${UCL_CHATGPT_REF}" \
    --model_result_path "${UCL_RES_PATH}" \
    --datasource_path "${BASE_DIR}/UCL-bench/dataset/legal_data_sample.json" \
    --result_path "${UCL_SCORE_PATH}" \
    --judge_model_name "${JUDGE_MODEL_NAME}" \
    --api_key "${JUDGE_API_KEY}" \
    --api_url "${JUDGE_API_URL}" \
    --workers ${WORKERS}

info " -> 3. 统计汇总"
python ${BASE_DIR}/searchr1-qwen3/bench/ucl/ucl_result.py \
    --score_path "${UCL_SCORE_PATH}" \
    --inference_path "${UCL_RES_PATH}" \
    --output_path "${UCL_RESULT_PATH}"

# ================= 5. 收尾清理 =================
info "========================================================="
info "🎉 评测任务圆满结束！报告已生成于："
info "   - UCL:      ${UCL_RESULT_PATH}"
info "========================================================="